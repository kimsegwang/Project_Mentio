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

[Memory Summarization 자동 트리거]
같은 순차 큐 안에서 신규 기억이 실제로 INSERT될 때마다 사용자별 누적 카운터를 올리고,
PROFILE_SUMMARY_TRIGGER_COUNT에 도달하거나(또는 기존 프로필 요약이 아예 없는 Cold Start
상태의 최초 적재 시점에) memory_service.summarize_user_profile()을 같은 스레드에서
호출한다. 별도 스케줄러 없이 기존 순차 큐에 얹는 방식이라 "대화 지연에 영향 없음"
원칙을 그대로 유지한다.
"""
import queue
import threading
import logging
from collections import defaultdict
from typing import Dict, Optional, Tuple

from config import settings
from server.repositories.memory_repository import get_profile_summary
from server.services import memory_service

logger = logging.getLogger(__name__)


class MemoryWriteWorker:
    def __init__(self):
        self._queue: "queue.Queue[Tuple[str, str]]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # [Memory Summarization] 사용자별 "직전 요약 이후 신규 적재된 기억" 누적 카운터.
        # 동일 스레드에서만 증감되므로 별도 락 없이 안전하다.
        self._new_memory_counts: Dict[str, int] = defaultdict(int)

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
                new_id = memory_service.extract_and_store(user_text, user_id)
                if new_id is not None:
                    self._handle_new_memory_inserted(user_id)
            except Exception as e:
                logger.warning(f"[MemoryWriteWorker] 기억 적재 처리 중 예외 발생: {e}")
            finally:
                self._queue.task_done()

    def _handle_new_memory_inserted(self, user_id: str) -> None:
        """
        [Memory Summarization] 신규 기억이 실제로 INSERT된 직후(Dedup 스킵 제외) 호출된다.
        누적 카운터가 PROFILE_SUMMARY_TRIGGER_COUNT에 도달했거나, 기존 프로필 요약이 아예
        없는 Cold Start 상태라면 같은 스레드에서 즉시 summarize_user_profile()을 트리거한다.
        어느 단계에서 예외가 발생해도 워커 스레드가 죽지 않도록 이 메서드 전체를 보호한다.
        """
        try:
            self._new_memory_counts[user_id] += 1
            count = self._new_memory_counts[user_id]

            should_summarize = count >= settings.PROFILE_SUMMARY_TRIGGER_COUNT

            if not should_summarize:
                try:
                    should_summarize = get_profile_summary(user_id=user_id) is None
                except Exception as e:
                    logger.warning(f"[MemoryWriteWorker] Cold Start 판별용 프로필 조회 실패: {e}")

            if not should_summarize:
                return

            logger.info(
                f"[MemoryWriteWorker] 프로필 요약 자동 트리거 발동 (user: {user_id}, 누적 신규 기억: {count}개)"
            )
            result = memory_service.summarize_user_profile(user_id=user_id)

            if result is not None:
                self._new_memory_counts[user_id] = 0
                logger.info(f"[MemoryWriteWorker] 프로필 요약 갱신 및 누적 카운터 리셋 완료 (user: {user_id})")
            else:
                logger.warning(
                    f"[MemoryWriteWorker] 프로필 요약 트리거되었으나 결과 없음, 카운터 유지 (user: {user_id})"
                )
        except Exception as e:
            logger.warning(f"[MemoryWriteWorker] 프로필 요약 자동 트리거 처리 중 예외 발생: {e}")


# Spring Bean 스타일 전역 싱글톤 등록
memory_write_worker = MemoryWriteWorker()
