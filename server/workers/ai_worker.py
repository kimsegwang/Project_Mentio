import queue
import threading
import time
from typing import Any, List, Tuple

from server.repositories.log_repository import insert_interaction_log
from server.schemas.action import RobotAction
from server.services.brain_service import BrainService


class AIWorker:
    def __init__(self, brain_service: BrainService):
        self.brain_service = brain_service
        # 과도한 요청 적체를 방지하기 위해 maxsize=1 적용
        self.request_queue: queue.Queue[Tuple[str, str, List[Any], float]] = queue.Queue(maxsize=1)
        self.response_queue: queue.Queue[Tuple[RobotAction, str, float]] = queue.Queue()
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        """백그라운드 데몬 스레드 구동"""
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
                # 0.2초 타임아웃으로 stop_event를 주기적으로 체크
                task = self.request_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            trigger_type, prompt_text, contents, req_time = task
            print(f"[AIWorker] '{trigger_type}' 추론 시작 (Background Thread)...")

            # 1. Gemini VLM / 텍스트 추론 (Service 호출)
            action = self.brain_service.infer_action(contents)
            latency = time.time() - req_time
            print(f"[AIWorker] '{trigger_type}' 추론 완료 ({latency:.2f}s) -> Action: {action.emotion}")

            # 2. SQLD 특화: 백그라운드에서 지연 없이 PostgreSQL DB에 로그 적재
            try:
                insert_interaction_log(
                    trigger_type=trigger_type,
                    prompt=prompt_text,
                    action=action,
                    latency_seconds=latency
                )
            except Exception as e:
                print(f"[AIWorker DB Warning] 로그 적재 중 예외 발생: {e}")

            # 3. 메인 스레드로 결과 전달
            self.response_queue.put((action, trigger_type, latency))
            self.request_queue.task_done()