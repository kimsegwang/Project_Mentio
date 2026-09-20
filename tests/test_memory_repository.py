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
from server.schemas.memory import MemoryRecord, UserProfileSummary


# fetchone_row을 넘기지 않은 기존 호출부(insert_memory의 RETURNING id 테스트)와
# "명시적으로 None(행 없음)을 반환"하는 신규 호출부(get_profile_summary)를 구분하기 위한 마커
_UNSET = object()


class FakeCursor:
    def __init__(self, rows=None, returning_id=None, fetchone_row=_UNSET):
        self._rows = rows or []
        self._returning_id = returning_id
        self._fetchone_row = fetchone_row
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

    def fetchone(self):
        if self._fetchone_row is not _UNSET:
            return self._fetchone_row
        return (self._returning_id,)


class FakeConnection:
    def __init__(self, rows=None, returning_id=None, fetchone_row=_UNSET):
        self.fake_cursor = FakeCursor(rows=rows, returning_id=returning_id, fetchone_row=fetchone_row)
        self.committed = False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        self.committed = True


def _patch_db(monkeypatch, rows=None, returning_id=None, fetchone_row=_UNSET):
    fake_conn = FakeConnection(rows=rows, returning_id=returning_id, fetchone_row=fetchone_row)

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


def test_search_filters_out_inactive_memories(monkeypatch):
    """무효화(is_active=FALSE)된 기억은 Dedup 검사/RAG 주입 어느 쪽에도 노출되면 안 된다."""
    fake_conn = _patch_db(monkeypatch, rows=[])

    memory_repository.search_similar_memories([0.1] * 384, user_id="primary_user")

    query = fake_conn.fake_cursor.executed_query
    assert "is_active = TRUE" in query


# --- insert_memory(): RETURNING id ---

def test_insert_memory_returns_new_id(monkeypatch):
    fake_conn = _patch_db(monkeypatch, returning_id=42)

    new_id = memory_repository.insert_memory(user_id="primary_user", fact_text="사과를 좋아해", embedding=[0.1] * 384)

    assert new_id == 42
    assert fake_conn.committed is True
    assert "RETURNING id" in fake_conn.fake_cursor.executed_query


def test_insert_memory_swallows_db_exception_and_returns_none(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(memory_repository, "get_db_connection", raise_error)

    result = memory_repository.insert_memory(user_id="primary_user", fact_text="사과를 좋아해", embedding=[0.1] * 384)

    assert result is None


# --- invalidate_memories(): 소프트 삭제(is_active=FALSE) ---

def test_invalidate_memories_updates_is_active_flag_with_superseded_by(monkeypatch):
    fake_conn = _patch_db(monkeypatch)

    memory_repository.invalidate_memories([1, 2], superseded_by=42)

    query, params = fake_conn.fake_cursor.executed_query, fake_conn.fake_cursor.executed_params
    assert "is_active = FALSE" in query
    assert "DELETE" not in query.upper()
    assert params == (42, [1, 2])
    assert fake_conn.committed is True


def test_invalidate_memories_noop_when_ids_empty(monkeypatch):
    fake_conn = _patch_db(monkeypatch)

    memory_repository.invalidate_memories([], superseded_by=42)

    assert fake_conn.fake_cursor.executed_query is None


def test_invalidate_memories_swallows_db_exception(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(memory_repository, "get_db_connection", raise_error)

    # 예외가 상위로 전파되지 않아야 한다 (대화 파이프라인에 영향 없음)
    memory_repository.invalidate_memories([1], superseded_by=42)


# --- get_active_memories(): [Memory Summarization] 소스 데이터 조회 ---

def test_get_active_memories_filters_by_active_flag_and_orders_recent_first(monkeypatch):
    fake_conn = _patch_db(monkeypatch, rows=[])

    memory_repository.get_active_memories(user_id="primary_user", limit=50)

    query, params = fake_conn.fake_cursor.executed_query, fake_conn.fake_cursor.executed_params
    assert "is_active = TRUE" in query
    assert "ORDER BY created_at DESC" in query
    assert params == ("primary_user", 50)


def test_get_active_memories_returns_rows_as_memory_records(monkeypatch):
    rows = [(1, "primary_user", "커피를 좋아한다", None)]
    _patch_db(monkeypatch, rows=rows)

    result = memory_repository.get_active_memories(user_id="primary_user")

    assert result == [
        MemoryRecord(id=1, user_id="primary_user", fact_text="커피를 좋아한다", created_at=None)
    ]


def test_get_active_memories_swallows_db_exception_and_returns_empty_list(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(memory_repository, "get_db_connection", raise_error)

    result = memory_repository.get_active_memories(user_id="primary_user")

    assert result == []


# --- upsert_profile_summary(): UPSERT ---

def test_upsert_profile_summary_upserts_with_on_conflict(monkeypatch):
    fake_conn = _patch_db(monkeypatch)

    memory_repository.upsert_profile_summary(
        user_id="primary_user", summary_text="커피를 좋아하는 애묘인", source_memory_count=3
    )

    query, params = fake_conn.fake_cursor.executed_query, fake_conn.fake_cursor.executed_params
    assert "ON CONFLICT (user_id) DO UPDATE" in query
    assert params == ("primary_user", "커피를 좋아하는 애묘인", 3)
    assert fake_conn.committed is True


def test_upsert_profile_summary_swallows_db_exception(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(memory_repository, "get_db_connection", raise_error)

    # 예외가 상위(MemoryService)로 전파되면 안 된다
    memory_repository.upsert_profile_summary(user_id="primary_user", summary_text="요약", source_memory_count=1)


# --- get_profile_summary(): 조회 ---

def test_get_profile_summary_returns_record_when_found(monkeypatch):
    from datetime import datetime

    updated_at = datetime(2026, 9, 20)
    _patch_db(monkeypatch, fetchone_row=("primary_user", "커피를 좋아하는 애묘인", 3, updated_at))

    result = memory_repository.get_profile_summary(user_id="primary_user")

    assert result == UserProfileSummary(
        user_id="primary_user", summary_text="커피를 좋아하는 애묘인", source_memory_count=3, updated_at=updated_at
    )


def test_get_profile_summary_returns_none_when_not_found(monkeypatch):
    _patch_db(monkeypatch, fetchone_row=None)

    result = memory_repository.get_profile_summary(user_id="primary_user")

    assert result is None


def test_get_profile_summary_swallows_db_exception_and_returns_none(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(memory_repository, "get_db_connection", raise_error)

    result = memory_repository.get_profile_summary(user_id="primary_user")

    assert result is None
