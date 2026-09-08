import queue
import threading
import time
from typing import Any, List, Tuple

from server.repositories.log_repository import insert_interaction_log
from server.repositories.preset_repository import load_emotion_presets, get_preset_for_emotion
from server.schemas.action import RobotAction, LLMResponse, FALLBACK_ACTION
from server.services.brain_service import BrainService


class AIWorker:
    def __init__(self, brain_service: BrainService):
        self.brain_service = brain_service
        self.request_queue: queue.Queue[Tuple[str, str, List[Any], float]] = queue.Queue(maxsize=1)
        self.response_queue: queue.Queue[Tuple[RobotAction, str, float]] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """백그라운드 데몬 스레드 구동 및 DB 프리셋 캐싱"""
        # 서버 시작 시 DB에서 감정 프리셋(RGB, duration)을 메모리에 1회 적재
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

    def _worker_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                task = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            trigger_type, prompt_text, contents, req_time = task
            print(f"[AIWorker] '{trigger_type}' 추론 시작 (Background Thread)...")

            # 1. Gemini VLM 경량 추론 (LLMResponse: emotion + speech)
            llm_response: LLMResponse = self.brain_service.infer_action(contents)
            latency = time.time() - req_time

            # 2. DB 프리셋 캐시에서 RGB 및 duration 매핑 -> RobotAction 조립
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

            # 3. PostgreSQL DB에 비동기 로그 적재
            try:
                insert_interaction_log(
                    trigger_type=trigger_type,
                    prompt=prompt_text,
                    action=action,
                    latency_seconds=latency
                )
            except Exception as e:
                print(f"[AIWorker DB Warning] 로그 적재 중 예외 발생: {e}")

            # 4. 메인 스레드로 완성된 RobotAction 전달
            self.response_queue.put((action, trigger_type, latency))
            self.request_queue.task_done()