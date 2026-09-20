"""
tests/test_memory_summary.py
Memory Summarization 파이프라인(MemoryService.summarize_user_profile)을
LLM/DB Mocking 기반으로 검증하는 pytest 단위 테스트.

검증 대상:
- 활성 기억이 없으면 LLM 호출/UPSERT 없이 None을 반환하고 조용히 생략한다
- 활성 기억이 있으면 fact_text만 추출해 BrainService.summarize_profile에 전달한다
- LLM 요약 결과를 user_profile_summary에 UPSERT하고(source_memory_count=활성 기억 개수)
  반환된 UserProfileSummary DTO에도 동일한 값이 반영된다
- LLM이 빈 문자열(공백 포함)을 반환하면 UPSERT를 스킵하고 None을 반환한다
- 조회/LLM/저장 어느 단계에서 예외가 발생해도 상위로 전파하지 않고 None으로 안전 폴백한다
  (배치/백그라운드 파이프라인이므로 실패가 대화 파이프라인에 영향을 주면 안 된다)
"""
import pytest

from config import settings
from server.schemas.memory import MemoryRecord, UserProfileSummary
from server.services import memory_service


def _make_memories(fact_texts):
    return [
        MemoryRecord(id=i, user_id="primary_user", fact_text=text, similarity=None, created_at=None)
        for i, text in enumerate(fact_texts, start=1)
    ]


# --- summarize_user_profile(): 활성 기억이 없는 경우 ---

def test_summarize_user_profile_skips_when_no_active_memories(monkeypatch):
    summarize_calls = []
    upsert_calls = []
    monkeypatch.setattr(memory_service, "get_active_memories", lambda **kwargs: [])
    monkeypatch.setattr(
        memory_service.brain_service,
        "summarize_profile",
        lambda fact_texts: summarize_calls.append(fact_texts) or "요약",
    )
    monkeypatch.setattr(memory_service, "upsert_profile_summary", lambda **kwargs: upsert_calls.append(kwargs))

    result = memory_service.summarize_user_profile(user_id="primary_user")

    assert result is None
    assert summarize_calls == []
    assert upsert_calls == []


# --- summarize_user_profile(): 정상 경로 ---

def test_summarize_user_profile_passes_fact_texts_only_to_brain_service(monkeypatch):
    memories = _make_memories(["커피를 좋아한다", "매일 아침 산책한다"])
    summarize_calls = []
    monkeypatch.setattr(memory_service, "get_active_memories", lambda **kwargs: memories)
    monkeypatch.setattr(
        memory_service.brain_service,
        "summarize_profile",
        lambda fact_texts: (summarize_calls.append(fact_texts), "커피를 좋아하고 아침형 인간이다")[1],
    )
    monkeypatch.setattr(memory_service, "upsert_profile_summary", lambda **kwargs: None)

    memory_service.summarize_user_profile(user_id="primary_user")

    assert summarize_calls == [["커피를 좋아한다", "매일 아침 산책한다"]]


def test_summarize_user_profile_upserts_generated_summary_with_source_count(monkeypatch):
    memories = _make_memories(["커피를 좋아한다", "매일 아침 산책한다", "고양이를 키운다"])
    upsert_calls = []
    monkeypatch.setattr(memory_service, "get_active_memories", lambda **kwargs: memories)
    monkeypatch.setattr(
        memory_service.brain_service, "summarize_profile", lambda fact_texts: "  커피를 좋아하는 애묘인  "
    )
    monkeypatch.setattr(memory_service, "upsert_profile_summary", lambda **kwargs: upsert_calls.append(kwargs))

    result = memory_service.summarize_user_profile(user_id="primary_user")

    assert len(upsert_calls) == 1
    assert upsert_calls[0]["user_id"] == "primary_user"
    # 앞뒤 공백은 저장 전 정리되어야 한다
    assert upsert_calls[0]["summary_text"] == "커피를 좋아하는 애묘인"
    assert upsert_calls[0]["source_memory_count"] == 3

    assert result == UserProfileSummary(
        user_id="primary_user", summary_text="커피를 좋아하는 애묘인", source_memory_count=3
    )


def test_summarize_user_profile_respects_source_limit_setting(monkeypatch):
    captured_kwargs = {}
    monkeypatch.setattr(
        memory_service,
        "get_active_memories",
        lambda **kwargs: (captured_kwargs.update(kwargs), [])[1],
    )

    memory_service.summarize_user_profile(user_id="primary_user")

    assert captured_kwargs["user_id"] == "primary_user"
    assert captured_kwargs["limit"] == settings.PROFILE_SUMMARY_SOURCE_LIMIT


# --- summarize_user_profile(): LLM 빈 응답 ---

@pytest.mark.parametrize("blank_summary", ["", "   ", "\n"])
def test_summarize_user_profile_skips_upsert_when_llm_returns_blank(monkeypatch, blank_summary):
    memories = _make_memories(["커피를 좋아한다"])
    upsert_calls = []
    monkeypatch.setattr(memory_service, "get_active_memories", lambda **kwargs: memories)
    monkeypatch.setattr(memory_service.brain_service, "summarize_profile", lambda fact_texts: blank_summary)
    monkeypatch.setattr(memory_service, "upsert_profile_summary", lambda **kwargs: upsert_calls.append(kwargs))

    result = memory_service.summarize_user_profile(user_id="primary_user")

    assert result is None
    assert upsert_calls == []


# --- summarize_user_profile(): 예외 안전성 ---

def test_summarize_user_profile_swallows_active_memory_lookup_exception(monkeypatch):
    def raise_error(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(memory_service, "get_active_memories", raise_error)

    result = memory_service.summarize_user_profile(user_id="primary_user")

    assert result is None


def test_summarize_user_profile_swallows_brain_service_exception(monkeypatch):
    memories = _make_memories(["커피를 좋아한다"])
    upsert_calls = []
    monkeypatch.setattr(memory_service, "get_active_memories", lambda **kwargs: memories)

    def raise_error(fact_texts):
        raise RuntimeError("llm down")

    monkeypatch.setattr(memory_service.brain_service, "summarize_profile", raise_error)
    monkeypatch.setattr(memory_service, "upsert_profile_summary", lambda **kwargs: upsert_calls.append(kwargs))

    result = memory_service.summarize_user_profile(user_id="primary_user")

    assert result is None
    assert upsert_calls == []


def test_summarize_user_profile_swallows_upsert_exception(monkeypatch):
    memories = _make_memories(["커피를 좋아한다"])
    monkeypatch.setattr(memory_service, "get_active_memories", lambda **kwargs: memories)
    monkeypatch.setattr(memory_service.brain_service, "summarize_profile", lambda fact_texts: "요약문")

    def raise_error(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(memory_service, "upsert_profile_summary", raise_error)

    # UPSERT가 실패해도 예외가 상위(호출부)로 전파되면 안 된다
    result = memory_service.summarize_user_profile(user_id="primary_user")

    assert result is None
