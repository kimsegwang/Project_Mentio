import queue
import threading
import time
import logging
from typing import Any, List, Tuple, Optional, Union
import numpy as np

from config import settings
from server.repositories.log_repository import insert_interaction_log
from server.repositories.preset_repository import load_emotion_presets, get_preset_for_emotion
from server.repositories.memory_repository import get_profile_summary, search_similar_memories
from server.schemas.action import RobotAction, LLMResponse, TriggerType
from server.services.brain_service import BrainService, brain_service
from server.services.stt_service import stt_service
from server.services.intent_service import intent_service
from server.services.embedding_service import embedding_service
from server.services.question_detector import question_detector
from server.services.speaker_service import SpeakerService, speaker_service
from server.workers.memory_write_worker import memory_write_worker
from server.services.tts_service import TTSService
from server.services.audio_player_service import AudioPlayerService
from server.adapters.audio_io import AudioSink

logger = logging.getLogger(__name__)


class AIWorker:
    def __init__(
        self,
        brain_service_instance: BrainService = brain_service,
        tts_service_instance: Optional[TTSService] = None,
        audio_player_instance: Optional[AudioSink] = None,
        speaker_service_instance: SpeakerService = speaker_service,
        on_task_completed: Optional[callable] = None, # 💡 콜백 주입받기
    ):
        self.brain_service = brain_service_instance
        self.tts_service = tts_service_instance or TTSService()
        self.audio_player = audio_player_instance or AudioPlayerService()
        self.speaker_service = speaker_service_instance
        self.on_task_completed = on_task_completed

        self.request_queue: queue.Queue[Tuple[str, str, List[Any], float]] = queue.Queue(maxsize=1)
        self.response_queue: queue.Queue[Tuple[RobotAction, str, float]] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """백그라운드 데몬 스레드 구동 및 DB 프리셋 캐싱"""
        load_emotion_presets()

        self.stop_event.clear()
        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()
        print("[AIWorker] 백그라운드 AI 추론 & DB 워커 스레드 시작.")

    def stop(self) -> None:
        """워커 스레드 안전 종료 신호"""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        print("[AIWorker] 워커 스레드 종료 완료.")

    def submit_task(self, trigger_type: str, prompt_text: str, contents: List[Any]) -> bool:
        """메인 루프에서 작업을 큐에 등록 (큐가 차있으면 무시하여 최신성 유지)"""
        try:
            current_time = time.time()
            self.request_queue.put_nowait((trigger_type, prompt_text, contents, current_time))
            return True
        except queue.Full:
            print("[AIWorker Warning] 이전 AI 작업이 진행 중이어서 신규 요청을 스킵합니다.")
            return False

    def poll_result(self) -> Tuple[RobotAction, str, float] | None:
        """메인 루프에서 AI 완료 결과를 논블로킹으로 확인"""
        try:
            return self.response_queue.get_nowait()
        except queue.Empty:
            return None

    def _play_speech_and_guard(self, speech_text: str) -> None:
        """공통 음성 합성 및 에코 캔슬링 블로킹 재생"""
        if not speech_text or not speech_text.strip():
            return
        
        try:
            print(f"🔊 [TTS 발화 시작] \"{speech_text}\"")
            audio_bytes = self.tts_service.synthesize(speech_text)
            if audio_bytes:
                self.audio_player.play_bytes_and_wait(audio_bytes)
                print("🔊 [TTS 발화 및 에코 잔향 가드 종료]")
        except Exception as e:
            print(f"[AIWorker TTS Warning] 음성 재생 실패: {e}")

    def _retrieve_memory_context(self, user_text: str, user_id: str = settings.DEFAULT_USER_ID) -> str:
        """
        [RAG Retrieval] 사용자 발화를 로컬 임베딩 후 pgvector에서 Top-K 유사 기억을 조회하여
        "[참고 기억] ..." 형태의 간결한 컨텍스트 문자열로 조립한다.
        [Memory Summarization] 여기에 더해 user_profile_summary에 저장된 고수준 페르소나 요약을
        "[사용자 프로필: ...]" 형태로 함께 주입해, 낱개 Top-K 기억만으로는 드러나지 않는
        사용자의 전반적인 성향/선호를 매 턴 저비용(단순 조회, LLM 재호출 없음)으로 반영한다.
        [다중 사용자 격리] user_id로 조회 범위를 한정해, 식별된 화자 본인의 프로필/기억만
        조회하고 다른 가족 구성원의 기억이 섞여 들어오지 않도록 한다.
        검색/조회 실패 시에도 대화 파이프라인이 끊기지 않도록 해당 구간만 생략한다.
        """
        context_lines = []

        try:
            profile = get_profile_summary(user_id=user_id)
            if profile and profile.summary_text:
                context_lines.append(f"[사용자 프로필: {profile.summary_text}]")
        except Exception as e:
            print(f"[AIWorker RAG Warning] 프로필 요약 조회 실패: {e}")

        try:
            query_embedding = embedding_service.embed(user_text)

            is_question = question_detector.is_question(user_text)
            threshold = (
                settings.RAG_QUESTION_SIMILARITY_THRESHOLD
                if is_question
                else settings.RAG_SIMILARITY_THRESHOLD
            )
            if is_question:
                print(f"🧠 [RAG] 의문문 감지 -> 완화된 임계값({threshold}) 적용")

            memories = search_similar_memories(
                query_embedding, user_id=user_id, top_k=settings.RAG_TOP_K, threshold=threshold
            )
        except Exception as e:
            print(f"[AIWorker RAG Warning] 장기 기억 검색 실패: {e}")
            return "\n".join(context_lines)

        if not memories:
            print(f"🧠 [RAG] 임계값({threshold}) 이상의 관련 기억 없음, 컨텍스트 주입 생략")
            return "\n".join(context_lines)

        for memory in memories:
            score = f"{memory.similarity:.2f}" if memory.similarity is not None else "N/A"
            print(f"🧠 [RAG] [참고 기억] {memory.fact_text} (유사도: {score})")

        context_lines.extend(f"[참고 기억] {memory.fact_text}" for memory in memories)
        return "\n".join(context_lines)

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            trigger_type, prompt_text, contents, req_time = task
            print(f"[AIWorker] '{trigger_type}' 추론 시작 (Background Thread)...")

            try:
                # 1. Gemini VLM 추론
                llm_response: LLMResponse = self.brain_service.infer_action(contents)
                latency = time.time() - req_time

                # 2. DB 프리셋 조립
                emotion_key = (
                    llm_response.emotion.value 
                    if hasattr(llm_response.emotion, "value") 
                    else str(llm_response.emotion)
                )
                preset = get_preset_for_emotion(emotion_key)

                action = RobotAction(
                    emotion=llm_response.emotion,
                    speech=llm_response.speech,
                    led_rgb=preset["rgb"],
                    duration=preset["duration"]
                )
                print(f"[AIWorker] '{trigger_type}' 매핑 완료 ({latency:.2f}s) -> Action: {emotion_key}, LED: {action.led_rgb}")

                # 3. DB 비동기 로깅
                try:
                    insert_interaction_log(
                        trigger_type=trigger_type,
                        prompt=prompt_text,
                        action=action,
                        latency_seconds=latency
                    )
                except Exception as e:
                    print(f"[AIWorker DB Warning] 로그 적재 실패: {e}")

                # 4. 메인 UI로 먼저 전달 -> 표정/LED 즉시 변경 (0초 체감)
                self.response_queue.put((action, trigger_type, latency))

                # 5. 스피커 음성 합성 및 에코 잔향 대기
                if action.speech:
                    self._play_speech_and_guard(action.speech)

            except Exception as e:
                print(f"[AIWorker Error] 작업 처리 중 예외 발생: {e}")
            finally:
                # 💡 [개선] 순환 import 없이 주입받은 콜백으로 안전하게 락 해제
                if self.on_task_completed:
                    self.on_task_completed()
                self.request_queue.task_done()

    def process_voice_interaction(
        self, 
        audio_data: Union[np.ndarray, bytes], 
        current_frame: Optional[Any] = None
    ) -> Optional[RobotAction]:
        total_start = time.time()

        # 0. 화자 식별 (VAD 직후, STT 이전) - 미등록 화자 전무/비활성화 시 자동 스킵(DEFAULT_USER_ID로 통과)
        #    [다중 사용자 격리] 여기서 판정된 user_id가 이후 RAG 검색/프로필 조회/장기 기억
        #    적재까지 일관되게 전달되어, 화자별 데이터가 서로 섞이지 않도록 한다.
        identification = self.speaker_service.identify_speaker(audio_data)
        if identification.similarity is not None:
            print(
                f"🔒 [AIWorker] 화자 식별 점수: {identification.similarity:.3f} "
                f"(기준: {identification.threshold:.3f}) -> "
                f"{'통과 (user: ' + str(identification.user_id) + ')' if identification.is_match else '실패 (차단)'}"
            )
        if not identification.is_match:
            print("[AIWorker] 화자 식별 실패 -> 등록되지 않은 화자로 판단, 파이프라인 진입을 차단합니다.")
            return None

        user_id = identification.user_id or settings.DEFAULT_USER_ID

        # 1. STT 변환
        t0 = time.time()
        user_text = stt_service.transcribe(audio_data)
        stt_latency = time.time() - t0

        if not user_text or not user_text.strip():
            print("[AIWorker] 인식된 음성 텍스트가 없습니다.")
            return None

        print(f"\n🎤 [STT 인식 결과] \"{user_text}\" (소요: {stt_latency:.2f}s)")

        # 2. 의도 판별 (VOICE_CHAT vs VOICE_VISION vs VOICE_TIME_RULE)
        t1 = time.time()
        trigger_str, needs_vision = intent_service.analyze_voice_intent(user_text)
        intent_latency = time.time() - t1

        if trigger_str == TriggerType.VOICE_TIME_RULE.value:
            # 2-1. [룰 기반 즉시 처리] 시간 질의는 Gemini 호출을 건너뛰고 로컬 시계로 즉답
            prompt_text = f"[룰 기반 즉시 처리] 시간 질의: \"{user_text}\""
            print(f"⏰ [Rule-based Instant] 시간 질의 감지 (Type: {trigger_str}) -> LLM 호출 스킵, 로컬 시계로 즉답합니다.")

            t2 = time.time()
            llm_response: LLMResponse = intent_service.build_time_response()
            gemini_latency = time.time() - t2
            rag_latency = 0.0
        else:
            # 2-2. [RAG Retrieval] 시간 룰 질의가 아닐 때만 장기 기억 검색 (Top-K=2)
            t_rag = time.time()
            memory_context = self._retrieve_memory_context(user_text, user_id=user_id)
            rag_latency = time.time() - t_rag
            if memory_context:
                print(f"🧠 [RAG] 장기 기억 컨텍스트 주입 ({rag_latency*1000:.1f}ms):\n{memory_context}")

            # 3. Contents Payload 조립
            contents = []
            if memory_context:
                contents.append(memory_context)

            if needs_vision and current_frame is not None:
                contents.append(current_frame)
                prompt_text = f"사용자의 시각 기반 질문: \"{user_text}\""
                print(f"📸 [Vision Pipeline] 시각 동봉 결정 (Type: {trigger_str}) -> 480p 스냅샷을 Gemini로 전송합니다.")
            else:
                prompt_text = f"사용자의 음성 대화: \"{user_text}\""
                print(f"💬 [Text Pipeline] 순수 텍스트 결정 (Type: {trigger_str}) -> 사진 제외, 텍스트만 전송합니다.")

            contents.append(prompt_text)

            # 4. Gemini 추론
            t2 = time.time()
            llm_response: LLMResponse = self.brain_service.infer_action(contents)
            gemini_latency = time.time() - t2

        total_latency = time.time() - total_start
        print(f"[⏱️ 속도 분석] 총 소요: {total_latency:.2f}s | STT: {stt_latency:.2f}s | Intent: {intent_latency*1000:.1f}ms | RAG: {rag_latency*1000:.1f}ms | Gemini: {gemini_latency:.2f}s")

        # 5. Emotion 매핑 및 RobotAction 조립
        emotion_key = (
            llm_response.emotion.value 
            if hasattr(llm_response.emotion, "value") 
            else str(llm_response.emotion)
        )
        preset = get_preset_for_emotion(emotion_key)

        action = RobotAction(
            emotion=llm_response.emotion,
            speech=llm_response.speech,
            led_rgb=preset["rgb"],
            duration=preset["duration"]
        )

        # 6. DB 로깅
        try:
            insert_interaction_log(
                trigger_type=trigger_str,
                prompt=prompt_text,
                action=action,
                latency_seconds=total_latency
            )
        except Exception as e:
            print(f"[AIWorker DB Warning] 로그 적재 실패: {e}")

        # 7. UI 반영을 위해 큐에 결과 즉각 전달
        self.response_queue.put((action, trigger_str, total_latency))

        # 8. [핵심 에코 방어] 음성 출력 완료 시점까지 handle_voice_interaction_thread를 블로킹
        #    이 작업이 끝나야 main.py의 finally 블록에서 is_processing = False가 호출됩니다.
        if action.speech:
            self._play_speech_and_guard(action.speech)

        # 9. [B-1] TTS 완료 후 규칙 기반 필터링 + 비동기 적재 (대화 지연 영향 0)
        #    MemoryWriteWorker의 순차 큐에 위임해, 연속 발화 시 모순 판정/무효화 순서가
        #    실제 발화 순서와 뒤바뀌는 경쟁 상태를 방지한다.
        if trigger_str != TriggerType.VOICE_TIME_RULE.value:
            memory_write_worker.submit(user_text, user_id=user_id)

        return action


# Spring Bean 스타일 전역 싱글톤 등록
ai_worker = AIWorker()