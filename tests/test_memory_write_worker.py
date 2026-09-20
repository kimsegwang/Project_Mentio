"""
tests/test_memory_write_worker.py
MemoryWriteWorker(단일 소비자 스레드 + queue.Queue) 검증 pytest 단위 테스트.

검증 대상:
- submit()은 즉시 반환되며(논블로킹) 실제 처리는 백그라운드 스레드에 위임한다
- 연속으로 submit된 요청은 큐에 들어간 순서 그대로(발화 순서대로) 직렬 처리된다
  (모순 해결/Invalidation 로직이 순서에 의존하므로 동시 실행으로 인한 경쟁 상태를 방지해야 함)
- extract_and_store 처리 중 예외가 발생해도 워커 스레드가 죽지 않고 다음 요청을 계속 처리한다
- [Memory Summarization 자동 트리거] 신규 기억이 실제로 INSERT(extract_and_store가 id 반환)될 때만
  누적 카운터가 오르고, PROFILE_SUMMARY_TRIGGER_COUNT 도달 또는 Cold Start(기존 프로필 없음) 시
  summarize_user_profile()이 같은 스레드에서 트리거되며, 성공 시에만 카운터가 리셋된다
"""
import time

import pytest

from config import settings
from server.schemas.memory import UserProfileSummary
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


def _wait_until_queue_drained(worker, timeout=2.0):
    """
    큐에 남은 아이템이 없는지(.empty())가 아니라, 제출된 모든 항목이 task_done()까지
    완료되었는지(unfinished_tasks == 0)를 기다린다. get()은 처리 시작 "전"에 큐를 비우므로
    .empty()만 보면 _handle_new_memory_inserted()가 아직 실행 중인데도 조기 통과할 수 있다.
    """
    return _wait_until(lambda: worker._queue.unfinished_tasks == 0, timeout=timeout)


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


# --- [Memory Summarization 자동 트리거] ---

def _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder):
    """extract_and_store가 실제 INSERT처럼 매 호출마다 증가하는 id를 반환하도록 스텁."""
    def fake_extract(user_text, user_id):
        next_id_holder[0] += 1
        return next_id_holder[0]

    monkeypatch.setattr(worker_module.memory_service, "extract_and_store", fake_extract)


def test_dedup_skip_does_not_increment_counter_or_trigger_summary(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    # 근접 중복으로 스킵된 경우 extract_and_store는 None을 반환한다
    monkeypatch.setattr(worker_module.memory_service, "extract_and_store", lambda user_text, user_id: None)
    monkeypatch.setattr(worker_module, "get_profile_summary", lambda user_id: None)
    summarize_calls = []
    monkeypatch.setattr(
        worker_module.memory_service, "summarize_user_profile", lambda user_id: summarize_calls.append(user_id)
    )

    for _ in range(settings.PROFILE_SUMMARY_TRIGGER_COUNT + 2):
        worker.submit("중복 발화", user_id="primary_user")

    # 모두 dedup 스킵되어 트리거될 일이 없어야 한다
    assert _wait_until_queue_drained(worker)
    time.sleep(0.05)
    assert summarize_calls == []


def test_new_memory_below_threshold_does_not_trigger_when_profile_already_exists(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    next_id_holder = [0]
    _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder)
    # Cold Start가 아님(이미 프로필 존재) -> 카운트가 임계값 미만이면 트리거되면 안 된다
    monkeypatch.setattr(
        worker_module,
        "get_profile_summary",
        lambda user_id: UserProfileSummary(user_id=user_id, summary_text="기존 요약", source_memory_count=10),
    )
    summarize_calls = []
    monkeypatch.setattr(
        worker_module.memory_service, "summarize_user_profile", lambda user_id: summarize_calls.append(user_id)
    )

    for _ in range(settings.PROFILE_SUMMARY_TRIGGER_COUNT - 1):
        worker.submit("신규 발화", user_id="primary_user")

    assert _wait_until_queue_drained(worker)
    time.sleep(0.05)
    assert summarize_calls == []


def test_new_memory_triggers_summary_at_threshold_and_resets_counter(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    next_id_holder = [0]
    _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder)
    monkeypatch.setattr(
        worker_module,
        "get_profile_summary",
        lambda user_id: UserProfileSummary(user_id=user_id, summary_text="기존 요약", source_memory_count=10),
    )
    summarize_calls = []
    monkeypatch.setattr(
        worker_module.memory_service,
        "summarize_user_profile",
        lambda user_id: (
            summarize_calls.append(user_id),
            UserProfileSummary(user_id=user_id, summary_text="새 요약", source_memory_count=5),
        )[1],
    )

    for _ in range(settings.PROFILE_SUMMARY_TRIGGER_COUNT):
        worker.submit("신규 발화", user_id="primary_user")

    assert _wait_until(lambda: len(summarize_calls) == 1)
    assert summarize_calls == ["primary_user"]
    # 카운터가 리셋되었으므로 트리거 직후 바로 다음 한 건으로는 재트리거되면 안 된다
    worker.submit("신규 발화", user_id="primary_user")
    assert _wait_until_queue_drained(worker)
    time.sleep(0.05)
    assert summarize_calls == ["primary_user"]


def test_cold_start_triggers_summary_immediately_before_threshold(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    next_id_holder = [0]
    _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder)
    # 기존 프로필이 아예 없는 Cold Start 상태
    monkeypatch.setattr(worker_module, "get_profile_summary", lambda user_id: None)
    summarize_calls = []
    monkeypatch.setattr(
        worker_module.memory_service,
        "summarize_user_profile",
        lambda user_id: (
            summarize_calls.append(user_id),
            UserProfileSummary(user_id=user_id, summary_text="최초 요약", source_memory_count=1),
        )[1],
    )

    # PROFILE_SUMMARY_TRIGGER_COUNT에 한참 못 미치는 단 1건만 적재해도 즉시 트리거되어야 한다
    worker.submit("첫 신규 발화", user_id="primary_user")

    assert _wait_until(lambda: len(summarize_calls) == 1)
    assert summarize_calls == ["primary_user"]


def test_counter_not_reset_when_summary_result_is_none(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    next_id_holder = [0]
    _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder)
    monkeypatch.setattr(
        worker_module,
        "get_profile_summary",
        lambda user_id: UserProfileSummary(user_id=user_id, summary_text="기존 요약", source_memory_count=10),
    )
    summarize_calls = []
    # 요약 실패(None 반환) 시나리오
    monkeypatch.setattr(
        worker_module.memory_service,
        "summarize_user_profile",
        lambda user_id: summarize_calls.append(user_id) or None,
    )

    for _ in range(settings.PROFILE_SUMMARY_TRIGGER_COUNT):
        worker.submit("신규 발화", user_id="primary_user")
    assert _wait_until(lambda: len(summarize_calls) == 1)

    # 카운터가 리셋되지 않았으므로 바로 다음 1건만 적재해도 재트리거되어야 한다
    worker.submit("신규 발화", user_id="primary_user")
    assert _wait_until(lambda: len(summarize_calls) == 2)


def test_worker_survives_exception_during_summary_trigger(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    next_id_holder = [0]
    _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder)
    monkeypatch.setattr(worker_module, "get_profile_summary", lambda user_id: None)

    def raise_error(user_id):
        raise RuntimeError("llm down")

    monkeypatch.setattr(worker_module.memory_service, "summarize_user_profile", raise_error)

    # Cold Start 조건으로 즉시 트리거되지만 내부에서 예외가 발생 -> 워커가 죽지 않고 다음 요청을 계속 처리해야 한다
    worker.submit("첫 신규 발화", user_id="primary_user")
    worker.submit("두 번째 발화", user_id="primary_user")

    assert _wait_until_queue_drained(worker)
    assert worker._thread.is_alive()


def test_worker_survives_exception_during_cold_start_lookup(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    next_id_holder = [0]
    _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder)

    def raise_error(user_id):
        raise RuntimeError("db down")

    monkeypatch.setattr(worker_module, "get_profile_summary", raise_error)
    summarize_calls = []
    monkeypatch.setattr(
        worker_module.memory_service, "summarize_user_profile", lambda user_id: summarize_calls.append(user_id)
    )

    worker.submit("신규 발화", user_id="primary_user")

    assert _wait_until_queue_drained(worker)
    assert worker._thread.is_alive()
    # Cold Start 판별 자체가 실패했으므로 안전하게 "요약 없음"으로 처리되어 트리거되지 않아야 한다
    time.sleep(0.05)
    assert summarize_calls == []


def test_counters_are_tracked_independently_per_user(worker, monkeypatch):
    import server.workers.memory_write_worker as worker_module

    next_id_holder = [0]
    _patch_extract_returns_id(monkeypatch, worker_module, next_id_holder)
    monkeypatch.setattr(
        worker_module,
        "get_profile_summary",
        lambda user_id: UserProfileSummary(user_id=user_id, summary_text="기존 요약", source_memory_count=10),
    )
    summarize_calls = []
    monkeypatch.setattr(
        worker_module.memory_service, "summarize_user_profile", lambda user_id: summarize_calls.append(user_id)
    )

    # user_a는 임계값 미만, user_b는 임계값 도달
    for _ in range(settings.PROFILE_SUMMARY_TRIGGER_COUNT - 1):
        worker.submit("신규 발화", user_id="user_a")
    for _ in range(settings.PROFILE_SUMMARY_TRIGGER_COUNT):
        worker.submit("신규 발화", user_id="user_b")

    assert _wait_until_queue_drained(worker)
    time.sleep(0.05)
    assert summarize_calls == ["user_b"]
