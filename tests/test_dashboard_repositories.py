"""
tests/test_dashboard_repositories.py
관리 대시보드(web/app.py) 전용 레포지토리 메서드 검증.

기존 레포지토리 테스트와 동일하게 실제 DB 없이 Fake 커넥션/커서로 SQL·파라미터 바인딩,
commit 여부, rowcount 기반 성공 판정, 예외 스월로우(가용성 우선) 동작을 검증한다.
"""
from contextlib import contextmanager
from datetime import datetime, timezone

import pytest

from server.repositories import memory_repository, speaker_repository


class FakeCursor:
    def __init__(self, rows=None, rowcount=1):
        self._rows = rows or []
        self.rowcount = rowcount
        self.executed_query = None
        self.executed_params = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    def execute(self, query, params=None):
        self.executed_query = query
        self.executed_params = params

    def fetchall(self):
        return self._rows


class FakeConnection:
    def __init__(self, rows=None, rowcount=1):
        self.fake_cursor = FakeCursor(rows=rows, rowcount=rowcount)
        self.committed = False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        self.committed = True


@pytest.fixture
def patch_db(monkeypatch):
    def _patch(module, rows=None, rowcount=1):
        fake_conn = FakeConnection(rows=rows, rowcount=rowcount)

        @contextmanager
        def fake_get_db_connection():
            yield fake_conn

        monkeypatch.setattr(module, "get_db_connection", fake_get_db_connection)
        return fake_conn

    return _patch


@pytest.fixture
def broken_db(monkeypatch):
    def _break(module):
        def raise_error():
            raise RuntimeError("db connection failed")

        monkeypatch.setattr(module, "get_db_connection", raise_error)

    return _break


# --- speaker_repository.list_speaker_profiles() ---

def test_list_speaker_profiles_maps_rows_without_embedding(patch_db):
    created = datetime(2026, 9, 1, tzinfo=timezone.utc)
    fake_conn = patch_db(speaker_repository, rows=[("dad", "아빠", True, created, created), ("mom", "엄마", False, None, None)])

    result = speaker_repository.list_speaker_profiles()

    query = fake_conn.fake_cursor.executed_query
    assert "speaker_embedding" not in query
    assert "WHERE" not in query
    assert [(s.user_id, s.display_name, s.is_active) for s in result] == [("dad", "아빠", True), ("mom", "엄마", False)]
    assert result[0].created_at == created


def test_list_speaker_profiles_active_only_filters_in_query(patch_db):
    fake_conn = patch_db(speaker_repository, rows=[])

    speaker_repository.list_speaker_profiles(include_inactive=False)

    assert "WHERE is_active = TRUE" in fake_conn.fake_cursor.executed_query


def test_list_speaker_profiles_swallows_db_exception(broken_db):
    broken_db(speaker_repository)
    assert speaker_repository.list_speaker_profiles() == []


# --- speaker_repository.update_display_name() ---

def test_update_display_name_binds_params_and_commits(patch_db):
    fake_conn = patch_db(speaker_repository, rowcount=1)

    assert speaker_repository.update_display_name("dad", "아버지") is True

    cursor = fake_conn.fake_cursor
    assert "SET display_name = %s" in cursor.executed_query
    assert cursor.executed_params == ("아버지", "dad")
    assert fake_conn.committed is True


def test_update_display_name_returns_false_when_user_missing(patch_db):
    patch_db(speaker_repository, rowcount=0)
    assert speaker_repository.update_display_name("ghost", "유령") is False


def test_update_display_name_swallows_db_exception(broken_db):
    broken_db(speaker_repository)
    assert speaker_repository.update_display_name("dad", "아버지") is False


# --- speaker_repository.set_speaker_active() ---

@pytest.mark.parametrize("is_active", [True, False])
def test_set_speaker_active_binds_flag(patch_db, is_active):
    fake_conn = patch_db(speaker_repository, rowcount=1)

    assert speaker_repository.set_speaker_active("mom", is_active) is True

    assert fake_conn.fake_cursor.executed_params == (is_active, "mom")
    assert "DELETE" not in fake_conn.fake_cursor.executed_query.upper()
    assert fake_conn.committed is True


def test_set_speaker_active_swallows_db_exception(broken_db):
    broken_db(speaker_repository)
    assert speaker_repository.set_speaker_active("mom", False) is False


# --- speaker_repository.delete_speaker_profile() ---

def test_delete_speaker_profile_only_touches_speaker_profiles(patch_db):
    fake_conn = patch_db(speaker_repository, rowcount=1)

    assert speaker_repository.delete_speaker_profile("dad") is True

    query = fake_conn.fake_cursor.executed_query
    assert "DELETE FROM speaker_profiles" in query
    # 장기 기억/프로필 요약은 보존되어야 한다
    assert "user_long_term_memory" not in query
    assert "user_profile_summary" not in query
    assert fake_conn.fake_cursor.executed_params == ("dad",)
    assert fake_conn.committed is True


def test_delete_speaker_profile_returns_false_when_user_missing(patch_db):
    patch_db(speaker_repository, rowcount=0)
    assert speaker_repository.delete_speaker_profile("ghost") is False


def test_delete_speaker_profile_swallows_db_exception(broken_db):
    broken_db(speaker_repository)
    assert speaker_repository.delete_speaker_profile("dad") is False


# --- memory_repository.count_active_memories_by_user() ---

def test_count_active_memories_by_user_groups_active_rows(patch_db):
    fake_conn = patch_db(memory_repository, rows=[("dad", 3), ("primary_user", 7)])

    result = memory_repository.count_active_memories_by_user()

    query = fake_conn.fake_cursor.executed_query
    assert "is_active = TRUE" in query
    assert "GROUP BY user_id" in query
    assert result == {"dad": 3, "primary_user": 7}


def test_count_active_memories_by_user_swallows_db_exception(broken_db):
    broken_db(memory_repository)
    assert memory_repository.count_active_memories_by_user() == {}


# --- memory_repository.soft_delete_memory() ---

def test_soft_delete_memory_deactivates_without_physical_delete(patch_db):
    fake_conn = patch_db(memory_repository, rowcount=1)

    assert memory_repository.soft_delete_memory(42) is True

    query = fake_conn.fake_cursor.executed_query
    assert "SET is_active = FALSE" in query
    assert "AND is_active = TRUE" in query
    assert "DELETE" not in query.upper()
    # 수동 삭제는 모순 해결(superseded_by)과 구분되어야 한다
    assert "superseded_by" not in query
    assert fake_conn.fake_cursor.executed_params == (42,)
    assert fake_conn.committed is True


def test_soft_delete_memory_returns_false_when_already_inactive(patch_db):
    patch_db(memory_repository, rowcount=0)
    assert memory_repository.soft_delete_memory(42) is False


def test_soft_delete_memory_swallows_db_exception(broken_db):
    broken_db(memory_repository)
    assert memory_repository.soft_delete_memory(42) is False
