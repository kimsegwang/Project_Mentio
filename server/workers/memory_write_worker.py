"""
server/workers/memory_write_worker.py
RAG 장기 기억 백그라운드 적재(memory_service.extract_and_store) 요청을
단일 소비자 스레드로 순차 처리하는 워커.

[왜 필요한가]
모순 해결(Invalidation) 로직이 도입되면서 memory_service.extract_and_store는
기존 기억 행을 is_active=FALSE로 무효화하는 부수효과를 갖게 되었다. 사용자가
빠르게 연속 발화("사과 좋아해" -> "역시 사과 싫어")할 때 턴마다 개별
threading.Thread를 던지면 두 백그라운드 스레드가 동시에 실행되어 무효화 순서가
사용자의 실제 발화 순서와 어긋날 수 있다(경쟁 상태). 이를 막기 위해 모든 기억
적재 요청을 큐에 넣고 단일 스레드가 발화 순서 그대로 하나씩 처리한다.

submit()은 queue.put_nowait로 즉시 반환되므로 대화 턴(TTS 완료 후 호출)에는
지연을 추가하지 않는다.
"""
import queue
import threading
import logging
from typing import Optional, Tuple

from config import settings
from server.services import memory_service

logger = logging.getLogger(__name__)


class MemoryWriteWorker:
    def __init__(self):
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """백그라운드 단일 소비자 스레드 구동"""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        print("[MemoryWriteWorker] 장기 기억 순차 적재 워커 스레드 시작.")

    def stop(self) -> None:
        """워커 스레드 안전 종료 신호"""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        print("[MemoryWriteWorker] 워커 스레드 종료 완료.")

    def submit(self, user_text: str, user_id: str = settings.DEFAULT_USER_ID) -> None:
        """
        대화 턴(TTS 완료 후 호출)에 영향 없는 논블로킹 큐잉.
        처리 순서 보장을 위해 호출부에서 별도 스레드를 직접 띄우지 않는다.
        """
        self._queue.put_nowait((user_text, user_id))

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                user_text, user_id = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                memory_service.extract_and_store(user_text, user_id)
            except Exception as e:
                logger.warning(f"[MemoryWriteWorker] 기억 적재 처리 중 예외 발생: {e}")
            finally:
                self._queue.task_done()


# Spring Bean 스타일 전역 싱글톤 등록
memory_write_worker = MemoryWriteWorker()
