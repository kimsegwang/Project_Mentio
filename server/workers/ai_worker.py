import queue
import threading
import time
import logging
from typing import Any, List, Tuple, Optional, Union
import numpy as np

from server.repositories.log_repository import insert_interaction_log
from server.repositories.preset_repository import load_emotion_presets, get_preset_for_emotion
from server.schemas.action import RobotAction, LLMResponse, TriggerType
from server.services.brain_service import BrainService, brain_service
from server.services.stt_service import stt_service
from server.services.intent_service import intent_service
from server.services.tts_service import TTSService
from server.services.audio_player_service import AudioPlayerService

logger = logging.getLogger(__name__)


class AIWorker:
    def __init__(
        self,
        brain_service_instance: BrainService = brain_service,
        tts_service_instance: Optional[TTSService] = None,
        audio_player_instance: Optional[AudioPlayerService] = None,
        on_task_completed: Optional[callable] = None, # 💡 콜백 주입받기
    ):
        self.brain_service = brain_service_instance
        self.tts_service = tts_service_instance or TTSService()
        self.audio_player = audio_player_instance or AudioPlayerService()
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

        # 1. STT 변환
        t0 = time.time()
        user_text = stt_service.transcribe(audio_data)
        stt_latency = time.time() - t0

        if not user_text or not user_text.strip():
            print("[AIWorker] 인식된 음성 텍스트가 없습니다.")
            return None

        print(f"\n🎤 [STT 인식 결과] \"{user_text}\" (소요: {stt_latency:.2f}s)")

        # 2. 의도 판별 (VOICE_CHAT vs VOICE_VISION)
        t1 = time.time()
        trigger_str, needs_vision = intent_service.analyze_voice_intent(user_text)
        intent_latency = time.time() - t1

        # 3. Contents Payload 조립
        contents = []
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
        print(f"[⏱️ 속도 분석] 총 소요: {total_latency:.2f}s | STT: {stt_latency:.2f}s | Intent: {intent_latency*1000:.1f}ms | Gemini: {gemini_latency:.2f}s")

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

        return action


# Spring Bean 스타일 전역 싱글톤 등록
ai_worker = AIWorker()