"""
tests/test_speaker_repository.py
speaker_profiles DB 접근(server/repositories/speaker_repository.py) 검증 pytest 단위 테스트.

memory_repository 테스트와 동일하게 실제 DB 없이 Fake 커넥션/커서로 SQL 및 파라미터
바인딩, 예외 스월로우(가용성 우선) 동작을 검증한다.
"""
from contextlib import contextmanager

import pytest

from server.repositories import speaker_repository


class FakeCursor:
    def __init__(self, rows=None):
        self._rows = rows or []
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
    def __init__(self, rows=None):
        self.fake_cursor = FakeCursor(rows=rows)
        self.committed = False

    def cursor(self):
        return self.fake_cursor

    def commit(self):
        self.committed = True


def _patch_db(monkeypatch, rows=None):
    fake_conn = FakeConnection(rows=rows)

    @contextmanager
    def fake_get_db_connection():
        yield fake_conn

    monkeypatch.setattr(speaker_repository, "get_db_connection", fake_get_db_connection)
    monkeypatch.setattr(speaker_repository, "register_vector", lambda conn: None)
    return fake_conn


# --- upsert_speaker_profile(): 신규 등록/갱신 UPSERT ---

def test_upsert_speaker_profile_upserts_with_on_conflict(monkeypatch):
    fake_conn = _patch_db(monkeypatch)

    success = speaker_repository.upsert_speaker_profile(
        user_id="dad", display_name="아빠", embedding=[0.1] * 256
    )

    query, params = fake_conn.fake_cursor.executed_query, fake_conn.fake_cursor.executed_params
    assert success is True
    assert "ON CONFLICT (user_id) DO UPDATE" in query
    assert params == ("dad", "아빠", [0.1] * 256)
    assert fake_conn.committed is True


def test_upsert_speaker_profile_swallows_db_exception_and_returns_false(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(speaker_repository, "get_db_connection", raise_error)

    result = speaker_repository.upsert_speaker_profile(user_id="dad", display_name="아빠", embedding=[0.1] * 256)

    assert result is False


# --- get_active_speaker_embeddings(): 1:N 비교 대상 캐시 소스 ---

def test_get_active_speaker_embeddings_filters_active_flag_in_query(monkeypatch):
    fake_conn = _patch_db(monkeypatch, rows=[])

    speaker_repository.get_active_speaker_embeddings()

    query = fake_conn.fake_cursor.executed_query
    assert "is_active = TRUE" in query


def test_get_active_speaker_embeddings_returns_records(monkeypatch):
    rows = [("dad", "아빠", [0.1] * 256), ("mom", "엄마", [0.2] * 256)]
    _patch_db(monkeypatch, rows=rows)

    result = speaker_repository.get_active_speaker_embeddings()

    assert len(result) == 2
    assert result[0].user_id == "dad"
    assert result[0].display_name == "아빠"
    assert result[0].embedding == [0.1] * 256
    assert result[1].user_id == "mom"


class FakePgvectorVector:
    """
    pgvector-python의 Vector 객체를 흉내낸 모의 클래스.
    register_vector() 등록 상태/버전에 따라 실제로 이런 객체가 조회 결과에 섞여 들어올 수 있어,
    speaker_repository가 이를 순수 list[float]로 정규화하는지 검증한다.
    """

    def __init__(self, values):
        self._values = list(values)

    def to_list(self):
        return self._values


def test_get_active_speaker_embeddings_normalizes_pgvector_vector_objects(monkeypatch):
    """
    실기 재현 버그: register_vector() 등록 여부에 따라 speaker_embedding 컬럼이 순수 list가
    아닌 pgvector.Vector 객체로 반환되면, 이를 그대로 넘길 경우 SpeakerService의 코사인 유사도
    계산에서 "unsupported operand type(s) for *: 'Vector' and 'Vector'" 타입 에러가 발생했다.
    리포지토리 경계에서 반드시 list[float]로 변환되어야 한다.
    """
    rows = [("dad", "아빠", FakePgvectorVector([0.1] * 256))]
    _patch_db(monkeypatch, rows=rows)

    result = speaker_repository.get_active_speaker_embeddings()

    assert result[0].embedding == [0.1] * 256
    assert isinstance(result[0].embedding, list)


def test_get_active_speaker_embeddings_normalizes_numpy_array_embedding(monkeypatch):
    import numpy as np

    rows = [("dad", "아빠", np.array([0.1] * 256, dtype=np.float32))]
    _patch_db(monkeypatch, rows=rows)

    result = speaker_repository.get_active_speaker_embeddings()

    assert isinstance(result[0].embedding, list)
    assert result[0].embedding == pytest.approx([0.1] * 256, abs=1e-6)


def test_get_active_speaker_embeddings_swallows_db_exception_and_returns_empty_list(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(speaker_repository, "get_db_connection", raise_error)

    result = speaker_repository.get_active_speaker_embeddings()

    assert result == []


# --- deactivate_speaker_profile(): 소프트 삭제 ---

def test_deactivate_speaker_profile_sets_is_active_false(monkeypatch):
    fake_conn = _patch_db(monkeypatch)

    speaker_repository.deactivate_speaker_profile("dad")

    query, params = fake_conn.fake_cursor.executed_query, fake_conn.fake_cursor.executed_params
    assert "is_active = FALSE" in query
    assert "DELETE" not in query.upper()
    assert params == ("dad",)
    assert fake_conn.committed is True


def test_deactivate_speaker_profile_swallows_db_exception(monkeypatch):
    def raise_error():
        raise RuntimeError("db connection failed")

    monkeypatch.setattr(speaker_repository, "get_db_connection", raise_error)

    # 예외가 상위로 전파되면 안 된다
    speaker_repository.deactivate_speaker_profile("dad")
