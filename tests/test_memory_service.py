"""
tests/test_memory_service.py
RAG 백그라운드 적재(B-1: 규칙 기반 필터링), 저장 전 근접 중복 방지(Deduplication),
모순 해결(Invalidation) 로직을 검증하는 pytest 단위 테스트.

CLAUDE.md/설계 결정 검증 대상:
- Gemini 재호출 없이 키워드/길이 휴리스틱만으로 "기억할 가치" 판별
- 필터 통과 시에만 로컬 임베딩 -> 근접 중복/모순 후보 검사 -> pgvector 적재 순서로 호출
- 근접 중복(RAG_DEDUP_SIMILARITY_THRESHOLD 이상) 발견 시 insert 없이 단순 스킵
- 모순 후보 구간(RAG_CONFLICT_CANDIDATE_THRESHOLD 이상 ~ Dedup 임계값 미만)에서만
  경량 LLM Reflection(brain_service.classify_memory_relation)을 호출
- CONTRADICTS 판정 시 신규 삽입 후 기존 기억을 invalidate_memories로 비활성화(soft delete)
- LLM이 후보 밖의 id를 반환해도(환각) 후보 id로 필터링해 안전하게 무시
- DB/임베딩 예외 발생 시에도 대화 파이프라인에 영향 없이 예외를 삼킴(swallow)
"""
import pytest

from config import settings
from server.schemas.memory import MemoryConflictResult, MemoryRecord, MemoryRelation
from server.services import memory_service


# --- is_memorable(): 규칙 기반 필터 ---

@pytest.mark.parametrize(
    "text",
    [
        "내 이름은 김세강이야",
        "나는 매일 아침 7시에 일어나",
        "나는 커피를 정말 좋아해",
        "다음 주 금요일에 발표가 있어",
        "이거 꼭 기억해줘",
    ],
)
def test_is_memorable_true_for_keyword_match(text: str):
    assert memory_service.is_memorable(text) is True


def test_is_memorable_true_for_long_context_without_keyword():
    text = "오늘 회사에서 있었던 일 때문에 조금 힘들었는데 그래도 잘 버텼어"
    assert len(text.strip()) >= memory_service.MIN_MEMORABLE_LENGTH
    assert memory_service.is_memorable(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "안녕",
        "고마워",
        "그렇구나",
        "응 맞아",
    ],
)
def test_is_memorable_false_for_short_generic_chat(text: str):
    assert memory_service.is_memorable(text) is False


@pytest.mark.parametrize("text", ["", "   ", None])
def test_is_memorable_false_for_empty_text(text):
    assert memory_service.is_memorable(text) is False


# --- extract_and_store(): 필터 통과 여부에 따른 임베딩/적재 호출 ---

def test_extract_and_store_skips_embed_and_insert_when_not_memorable(monkeypatch):
    embed_calls = []
    insert_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: embed_calls.append(text))
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: insert_calls.append(kwargs))

    memory_service.extract_and_store("안녕")

    assert embed_calls == []
    assert insert_calls == []


def test_extract_and_store_calls_embed_and_insert_when_memorable(monkeypatch):
    embed_calls = []
    insert_calls = []
    dummy_vector = [0.1] * 384
    monkeypatch.setattr(
        memory_service.embedding_service,
        "embed",
        lambda text: (embed_calls.append(text), dummy_vector)[1],
    )
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [])
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: insert_calls.append(kwargs))

    memory_service.extract_and_store("내 이름은 김세강이야", user_id="primary_user")

    assert embed_calls == ["내 이름은 김세강이야"]
    assert len(insert_calls) == 1
    assert insert_calls[0]["user_id"] == "primary_user"
    assert insert_calls[0]["fact_text"] == "내 이름은 김세강이야"
    assert insert_calls[0]["embedding"] == dummy_vector


def test_extract_and_store_swallows_embedding_exception(monkeypatch):
    def raise_error(text):
        raise RuntimeError("embedding backend down")

    insert_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", raise_error)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [])
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: insert_calls.append(kwargs))

    # 예외가 상위로 전파되지 않아야 한다 (대화 파이프라인에 영향 없음)
    memory_service.extract_and_store("내 이름은 김세강이야")

    assert insert_calls == []


def test_extract_and_store_swallows_insert_exception(monkeypatch):
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: [0.0] * 384)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [])

    def raise_error(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(memory_service, "insert_memory", raise_error)

    # insert_memory가 실패해도 예외가 상위로 전파되면 안 된다
    memory_service.extract_and_store("내 이름은 김세강이야")


# --- extract_and_store(): 저장 전 근접 중복(Deduplication) 검사 ---

def test_extract_and_store_skips_insert_when_duplicate_found(monkeypatch):
    dummy_vector = [0.2] * 384
    existing = MemoryRecord(
        id=1, user_id="primary_user", fact_text="나는 커피를 정말 좋아해", similarity=0.97, created_at=None
    )
    insert_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: dummy_vector)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [existing])
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: insert_calls.append(kwargs))

    memory_service.extract_and_store("나는 커피를 정말 좋아해", user_id="primary_user")

    assert insert_calls == []


def test_extract_and_store_fetches_nearest_neighbor_unconditionally(monkeypatch):
    """임계값 튜닝 가시성 + 모순 후보 확보를 위해 threshold=0.0/top_k=RAG_CONFLICT_TOP_K로 항상 조회해야 한다."""
    dummy_vector = [0.3] * 384
    search_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: dummy_vector)

    def fake_search(embedding, **kwargs):
        search_calls.append(kwargs)
        return []

    monkeypatch.setattr(memory_service, "search_similar_memories", fake_search)
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: None)

    memory_service.extract_and_store("나는 커피를 정말 좋아해", user_id="primary_user")

    assert len(search_calls) == 1
    assert search_calls[0]["user_id"] == "primary_user"
    assert search_calls[0]["top_k"] == settings.RAG_CONFLICT_TOP_K
    assert search_calls[0]["threshold"] == 0.0


def test_extract_and_store_inserts_when_similarity_below_dedup_threshold(monkeypatch):
    """중복 판정 자체는 서비스 계층에서 RAG_DEDUP_SIMILARITY_THRESHOLD와 직접 비교해야 한다."""
    dummy_vector = [0.4] * 384
    near_miss = MemoryRecord(
        id=2,
        user_id="primary_user",
        fact_text="커피 좋아해",
        similarity=settings.RAG_DEDUP_SIMILARITY_THRESHOLD - 0.01,
        created_at=None,
    )
    insert_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: dummy_vector)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [near_miss])
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: insert_calls.append(kwargs))
    # near_miss(0.84)는 RAG_CONFLICT_CANDIDATE_THRESHOLD(0.55) 이상이라 모순 후보에 해당하므로,
    # 실제 Gemini 호출 없이 결과를 강제해 순수 서비스 로직만 검증한다.
    monkeypatch.setattr(
        memory_service.brain_service,
        "classify_memory_relation",
        lambda new_fact, candidates: MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[]),
    )

    memory_service.extract_and_store("나는 커피를 정말 좋아해", user_id="primary_user")

    assert len(insert_calls) == 1


# --- extract_and_store(): 모순 해결(Invalidation) ---

def test_extract_and_store_calls_llm_reflection_only_for_conflict_band(monkeypatch):
    """RAG_CONFLICT_CANDIDATE_THRESHOLD 미만인 후보는 LLM 호출 없이(모순 판정 스킵) 바로 삽입되어야 한다."""
    dummy_vector = [0.5] * 384
    unrelated = MemoryRecord(
        id=3,
        user_id="primary_user",
        fact_text="오늘 날씨가 맑다",
        similarity=settings.RAG_CONFLICT_CANDIDATE_THRESHOLD - 0.01,
        created_at=None,
    )
    classify_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: dummy_vector)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [unrelated])
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: 99)
    monkeypatch.setattr(
        memory_service.brain_service,
        "classify_memory_relation",
        lambda new_fact, candidates: classify_calls.append((new_fact, candidates)),
    )

    memory_service.extract_and_store("나는 사과를 좋아해", user_id="primary_user")

    assert classify_calls == []


def test_extract_and_store_invalidates_conflicting_memory_on_contradiction(monkeypatch):
    """CONTRADICTS 판정 시: 신규 사실을 먼저 삽입하고, 그 id를 superseded_by로 기존 후보를 무효화해야 한다."""
    dummy_vector = [0.6] * 384
    conflicting = MemoryRecord(
        id=5, user_id="primary_user", fact_text="사과를 좋아해", similarity=0.70, created_at=None
    )
    insert_calls = []
    invalidate_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: dummy_vector)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [conflicting])
    monkeypatch.setattr(
        memory_service, "insert_memory", lambda **kwargs: (insert_calls.append(kwargs), 42)[1]
    )
    monkeypatch.setattr(
        memory_service,
        "invalidate_memories",
        lambda ids, superseded_by: invalidate_calls.append((ids, superseded_by)),
    )
    monkeypatch.setattr(
        memory_service.brain_service,
        "classify_memory_relation",
        lambda new_fact, candidates: MemoryConflictResult(relation=MemoryRelation.CONTRADICTS, conflicting_ids=[5]),
    )

    memory_service.extract_and_store("이제 사과 말고 포도가 좋아해", user_id="primary_user")

    assert len(insert_calls) == 1
    assert insert_calls[0]["fact_text"] == "이제 사과 말고 포도가 좋아해"
    assert invalidate_calls == [([5], 42)]


def test_extract_and_store_ignores_hallucinated_conflicting_ids(monkeypatch):
    """LLM이 후보 목록 밖의 id를 반환해도(환각) 실제 후보 id로 필터링해 무효화하지 않아야 한다."""
    dummy_vector = [0.7] * 384
    conflicting = MemoryRecord(
        id=5, user_id="primary_user", fact_text="사과를 좋아해", similarity=0.70, created_at=None
    )
    invalidate_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: dummy_vector)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [conflicting])
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: 42)
    monkeypatch.setattr(
        memory_service,
        "invalidate_memories",
        lambda ids, superseded_by: invalidate_calls.append((ids, superseded_by)),
    )
    monkeypatch.setattr(
        memory_service.brain_service,
        "classify_memory_relation",
        # 후보에 없는 id(999)를 반환하는 환각 상황을 시뮬레이션
        lambda new_fact, candidates: MemoryConflictResult(relation=MemoryRelation.CONTRADICTS, conflicting_ids=[999]),
    )

    memory_service.extract_and_store("이제 사과 말고 포도가 좋아해", user_id="primary_user")

    assert invalidate_calls == []


def test_extract_and_store_no_invalidation_when_relation_is_new(monkeypatch):
    """모순 후보가 있어도 LLM이 NEW로 판정하면 무효화 없이 신규 삽입만 되어야 한다."""
    dummy_vector = [0.8] * 384
    candidate = MemoryRecord(
        id=7, user_id="primary_user", fact_text="포도를 좋아해", similarity=0.70, created_at=None
    )
    insert_calls = []
    invalidate_calls = []
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: dummy_vector)
    monkeypatch.setattr(memory_service, "search_similar_memories", lambda *args, **kwargs: [candidate])
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: insert_calls.append(kwargs))
    monkeypatch.setattr(
        memory_service,
        "invalidate_memories",
        lambda ids, superseded_by: invalidate_calls.append((ids, superseded_by)),
    )
    monkeypatch.setattr(
        memory_service.brain_service,
        "classify_memory_relation",
        lambda new_fact, candidates: MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[]),
    )

    memory_service.extract_and_store("나는 사과를 좋아해", user_id="primary_user")

    assert len(insert_calls) == 1
    assert invalidate_calls == []
