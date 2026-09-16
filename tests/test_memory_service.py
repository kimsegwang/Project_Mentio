"""
tests/test_memory_service.py
RAG 백그라운드 적재(B-1: 규칙 기반 필터링) 로직을 검증하는 pytest 단위 테스트.

CLAUDE.md/설계 결정 검증 대상:
- Gemini 재호출 없이 키워드/길이 휴리스틱만으로 "기억할 가치" 판별
- 필터 통과 시에만 로컬 임베딩 -> pgvector 적재 순서로 호출
- DB/임베딩 예외 발생 시에도 대화 파이프라인에 영향 없이 예외를 삼킴(swallow)
"""
import pytest

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
    monkeypatch.setattr(memory_service, "insert_memory", lambda **kwargs: insert_calls.append(kwargs))

    # 예외가 상위로 전파되지 않아야 한다 (대화 파이프라인에 영향 없음)
    memory_service.extract_and_store("내 이름은 김세강이야")

    assert insert_calls == []


def test_extract_and_store_swallows_insert_exception(monkeypatch):
    monkeypatch.setattr(memory_service.embedding_service, "embed", lambda text: [0.0] * 384)

    def raise_error(**kwargs):
        raise RuntimeError("db down")

    monkeypatch.setattr(memory_service, "insert_memory", raise_error)

    # insert_memory가 실패해도 예외가 상위로 전파되면 안 된다
    memory_service.extract_and_store("내 이름은 김세강이야")
