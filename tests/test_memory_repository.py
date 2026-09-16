"""
tests/test_memory_repository.py
RAG 검색 쿼리(search_similar_memories)의 유사도 임계값(Threshold) 필터링을
검증하는 pytest 단위 테스트.

검증 대상:
- WHERE 절에 유사도 임계값 조건이 바인딩 파라미터와 함께 반영되는가
- threshold 인자를 생략하면 settings.RAG_SIMILARITY_THRESHOLD가 기본값으로 쓰이는가
- threshold 인자를 명시하면 해당 값이 쿼리 파라미터에 그대로 전달되는가
- top_k(LIMIT)는 threshold와 무관하게 그대로 유지되는가 (Top-K 상한 가드레일 보존)
- DB 조회 실패 시 예외를 삼키고 빈 리스트를 반환하는가 (기존 동작 회귀 확인)
"""
from contextlib import contextmanager

import pytest

from config import settings
from server.repositories import memory_repository
from server.schemas.memory import MemoryRecord


class FakeCursor:
    def __init__(self, rows):
        self._rows = rows
        self.executed_query = None
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, query, params):
        self.executed_query = query
        self.executed_params = params

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows):
        self.fake_cursor = FakeCursor(rows)

    def cursor(self):
        return self.fake_cursor


def _patch_db(monkeypatch, rows):
    fake_conn = FakeConnection(rows)

    @contextmanager
    def fake_get_db_connection():
        yield fake_conn

    monkeypatch.setattr(memory_repository, "get_db_connection", fake_get_db_connection)
    monkeypatch.setattr(memory_repository, "register_vector", lambda conn: None)
    return fake_conn


def test_search_uses_default_threshold_from_settings(monkeypatch):
    fake_conn = _patch_db(monkeypatch, rows=[])

    memory_repository.search_similar_memories([0.1] * 384, user_id="primary_user")

    query, params = fake_conn.fake_cursor.executed_query, fake_conn.fake_cursor.executed_params
    assert "AND 1 - (embedding <=> %s::vector) >= %s" in query
    assert settings.RAG_SIMILARITY_THRESHOLD in params


def test_search_uses_explicit_threshold_override(monkeypatch):
    fake_conn = _patch_db(monkeypatch, rows=[])

    memory_repository.search_similar_memories([0.1] * 384, user_id="primary_user", threshold=0.9)

    params = fake_conn.fake_cursor.executed_params
    assert 0.9 in params
    assert settings.RAG_SIMILARITY_THRESHOLD not in params or 0.9 != settings.RAG_SIMILARITY_THRESHOLD


def test_search_keeps_top_k_limit_alongside_threshold(monkeypatch):
    fake_conn = _patch_db(monkeypatch, rows=[])

    memory_repository.search_similar_memories([0.1] * 384, user_id="primary_user", top_k=2, threshold=0.65)

    params = fake_conn.fake_cursor.executed_params
    assert params[-1] == 2  # LIMIT 파라미터가 그대로 마지막에 바인딩됨


def test_search_returns_rows_as_memory_records(monkeypatch):
    rows = [(1, "primary_user", "커피를 좋아한다", None, 0.91)]
    _patch_db(monkeypatch, rows=rows)

    result = memory_repository.search_similar_memories([0.1] * 384)

    assert result == [
        MemoryRecord(id=1, user_id="primary_user", fact_text="커피를 좋아한다", created_at=None, similarity=0.91)
    ]


def test_search_returns_empty_list_when_below_threshold(monkeypatch):
    # 실제 필터링은 DB(WHERE 절)에서 수행되므로, 임계값 미만 행은 애초에 rows로 넘어오지 않는다.
    _patch_db(monkeypatch, rows=[])

    result = memory_repository.search_similar_memories([0.1] * 384)

    assert result == []


def test_search_swallows_db_exception_and_returns_empty_list(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(memory_repository, "get_db_connection", raise_error)

    result = memory_repository.search_similar_memories([0.1] * 384)

    assert result == []
