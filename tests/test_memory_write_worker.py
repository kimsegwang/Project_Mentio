"""
tests/test_memory_write_worker.py
MemoryWriteWorker(단일 소비자 스레드 + queue.Queue) 검증 pytest 단위 테스트.

검증 대상:
- submit()은 즉시 반환되며(논블로킹) 실제 처리는 백그라운드 스레드에 위임한다
- 연속으로 submit된 요청은 큐에 들어간 순서 그대로(발화 순서대로) 직렬 처리된다
  (모순 해결/Invalidation 로직이 순서에 의존하므로 동시 실행으로 인한 경쟁 상태를 방지해야 함)
- extract_and_store 처리 중 예외가 발생해도 워커 스레드가 죽지 않고 다음 요청을 계속 처리한다
"""
import time

import pytest

from server.workers.memory_write_worker import MemoryWriteWorker


@pytest.fixture()
def worker():
    w = MemoryWriteWorker()
    w.start()
    yield w
    w.stop()


def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_submit_processes_via_background_thread(worker, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "server.workers.memory_write_worker.memory_service.extract_and_store",
        lambda user_text, user_id: calls.append((user_text, user_id)),
    )

    worker.submit("내 이름은 김세강이야", user_id="primary_user")

    assert _wait_until(lambda: len(calls) == 1)
    assert calls[0] == ("내 이름은 김세강이야", "primary_user")


def test_submit_processes_requests_in_order(worker, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "server.workers.memory_write_worker.memory_service.extract_and_store",
        lambda user_text, user_id: calls.append(user_text),
    )

    worker.submit("사과 좋아해")
    worker.submit("역시 사과 싫어")
    worker.submit("포도 좋아해")

    assert _wait_until(lambda: len(calls) == 3)
    assert calls == ["사과 좋아해", "역시 사과 싫어", "포도 좋아해"]


def test_worker_survives_exception_and_continues_processing(worker, monkeypatch):
    calls = []

    def flaky_extract(user_text, user_id):
        if user_text == "boom":
            raise RuntimeError("db down")
        calls.append(user_text)

    monkeypatch.setattr(
        "server.workers.memory_write_worker.memory_service.extract_and_store", flaky_extract
    )

    worker.submit("boom")
    worker.submit("정상 발화")

    assert _wait_until(lambda: calls == ["정상 발화"])
