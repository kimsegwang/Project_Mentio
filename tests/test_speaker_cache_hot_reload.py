"""
tests/test_speaker_cache_hot_reload.py
대시보드 화자 변경 핫 리로드(SpeakerCacheSyncWorker + SpeakerService.refresh_speaker_profiles)
검증 pytest 단위 테스트.

- 변경 지문(MAX(updated_at), 행 수, 활성 행 수) 변화 감지 시에만 캐시 재적재
- 발화 처리 중 / 음성 온보딩 진행 중에는 갱신 보류 후 다음 주기에 재시도
- 재적재 결과 불일치(조회 실패로 빈 목록 등) 시 기존 캐시 유지
- 재적재 도중 온보딩 완료 무효화가 끼어들면 오래된 결과 폐기
- 비활성화/삭제된 화자의 세션 소프트패스 해제, 호칭 변경 반영
실제 DB 없이 get_active_speaker_embeddings와 지문 로더를 가짜 함수로 대체한다.
"""
from datetime import datetime
from unittest.mock import Mock

import numpy as np
import pytest

import server.services.speaker_service as speaker_service_module
from server.repositories.speaker_repository import SpeakerEmbeddingRecord
from server.services.speaker_service import SpeakerService
from server.workers.speaker_cache_sync_worker import SpeakerCacheSyncWorker

T0 = datetime(2026, 9, 28, 12, 0, 0)
T1 = datetime(2026, 9, 28, 12, 0, 30)


def _record(user_id, display_name, embedding=(1.0, 0.0)):
    return SpeakerEmbeddingRecord(user_id=user_id, display_name=display_name, embedding=list(embedding))


class FakeDb:
    """지문과 활성 프로필 목록을 함께 들고 있는 가짜 speaker_profiles 테이블."""

    def __init__(self, fingerprint, profiles):
        self.fingerprint = fingerprint
        self.profiles = profiles
        self.load_calls = 0

    def load_fingerprint(self):
        return self.fingerprint

    def load_profiles(self):
        self.load_calls += 1
        return list(self.profiles)


@pytest.fixture()
def db(monkeypatch):
    fake_db = FakeDb(fingerprint=(T0, 1, 1), profiles=[_record("dad", "아빠")])
    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", fake_db.load_profiles)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", True)
    return fake_db


@pytest.fixture()
def service():
    return SpeakerService(similarity_threshold=0.65)


@pytest.fixture()
def enrollment():
    fake = Mock()
    fake.is_active.return_value = False
    return fake


def _make_worker(db, service, enrollment, is_busy=lambda: False):
    worker = SpeakerCacheSyncWorker(
        speaker_service_instance=service,
        enrollment_service_instance=enrollment,
        is_busy=is_busy,
        poll_interval_sec=3600,
        fingerprint_loader=db.load_fingerprint,
    )
    worker._last_fingerprint = db.load_fingerprint()  # start()의 기준 지문 기록과 동일
    return worker


def _cached_names(service):
    return [(p.user_id, p.display_name) for p in service._load_active_profiles()]


# --- 변경 감지 ---

def test_poll_does_nothing_when_fingerprint_unchanged(db, service, enrollment):
    worker = _make_worker(db, service, enrollment)

    assert worker.poll_once() is False
    assert db.load_calls == 0


def test_poll_reloads_cache_when_display_name_changed(db, service, enrollment):
    worker = _make_worker(db, service, enrollment)
    assert _cached_names(service) == [("dad", "아빠")]

    db.profiles = [_record("dad", "아버지")]
    db.fingerprint = (T1, 1, 1)

    assert worker.poll_once() is True
    assert _cached_names(service) == [("dad", "아버지")]
    # 이후 같은 지문에서는 다시 조회하지 않는다
    calls = db.load_calls
    assert worker.poll_once() is False
    assert db.load_calls == calls


def test_poll_detects_hard_delete_without_updated_at_change(db, service, enrollment):
    """물리 DELETE는 MAX(updated_at)를 올리지 않으므로 행 수 변화로 감지해야 한다."""
    db.fingerprint = (T0, 2, 2)
    db.profiles = [_record("dad", "아빠"), _record("mom", "엄마")]
    worker = _make_worker(db, service, enrollment)
    assert len(_cached_names(service)) == 2

    db.profiles = [_record("dad", "아빠")]
    db.fingerprint = (T0, 1, 1)  # 삭제된 행이 MAX가 아니었던 경우

    assert worker.poll_once() is True
    assert _cached_names(service) == [("dad", "아빠")]


def test_poll_reload_happens_off_the_audio_path(db, service, enrollment, monkeypatch):
    """재적재가 폴러 스레드에서 끝나므로 이후 identify_speaker()는 DB를 조회하지 않는다."""
    worker = _make_worker(db, service, enrollment)
    db.profiles = [_record("dad", "아버지")]
    db.fingerprint = (T1, 1, 1)
    worker.poll_once()
    calls = db.load_calls

    monkeypatch.setattr(service, "preprocess", lambda audio, sample_rate=16000: audio)
    monkeypatch.setattr(service, "embed_preprocessed", lambda wav: np.array([1.0, 0.0]))
    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert result.display_name == "아버지"
    assert db.load_calls == calls


def test_poll_ignores_fingerprint_query_failure(db, service, enrollment):
    worker = _make_worker(db, service, enrollment)
    db.fingerprint = None

    assert worker.poll_once() is False
    assert db.load_calls == 0


# --- 안전성: 보류 조건 ---

def test_poll_defers_while_robot_is_busy_then_retries(db, service, enrollment):
    busy = {"value": True}
    worker = _make_worker(db, service, enrollment, is_busy=lambda: busy["value"])
    _cached_names(service)

    db.profiles = [_record("dad", "아버지")]
    db.fingerprint = (T1, 1, 1)

    assert worker.poll_once() is False
    assert _cached_names(service) == [("dad", "아빠")]

    busy["value"] = False
    assert worker.poll_once() is True
    assert _cached_names(service) == [("dad", "아버지")]


def test_poll_defers_while_voice_enrollment_in_progress(db, service, enrollment):
    worker = _make_worker(db, service, enrollment)
    _cached_names(service)
    enrollment.is_active.return_value = True

    db.profiles = [_record("dad", "아버지")]
    db.fingerprint = (T1, 1, 1)

    assert worker.poll_once() is False
    assert _cached_names(service) == [("dad", "아빠")]

    enrollment.is_active.return_value = False
    assert worker.poll_once() is True
    assert _cached_names(service) == [("dad", "아버지")]


# --- 안전성: 재적재 실패/경쟁 ---

def test_refresh_keeps_cache_when_loaded_count_mismatches(db, service, enrollment):
    """조회 실패로 빈 목록이 오면(식별 무검증 통과 위험) 기존 캐시를 유지하고 재시도한다."""
    worker = _make_worker(db, service, enrollment)
    _cached_names(service)

    db.profiles = []  # get_active_speaker_embeddings()가 예외를 삼키고 [] 반환한 상황
    db.fingerprint = (T1, 1, 1)

    assert worker.poll_once() is False
    assert _cached_names(service) == [("dad", "아빠")]

    db.profiles = [_record("dad", "아버지")]
    assert worker.poll_once() is True
    assert _cached_names(service) == [("dad", "아버지")]


def test_refresh_discards_stale_result_when_invalidated_during_load(db, service, monkeypatch):
    """재적재 조회 도중 온보딩 완료(reload_speaker_profiles)가 끼어들면 오래된 결과로 덮어쓰지 않는다."""

    def load_then_invalidate():
        stale = [_record("dad", "아빠")]
        service.reload_speaker_profiles()  # 조회 도중 다른 스레드에서 무효화 발생
        return stale

    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", load_then_invalidate)

    assert service.refresh_speaker_profiles(expected_active_count=1) is False
    assert service._profiles_loaded is False  # 다음 조회 시 최신 DB에서 지연 로딩


# --- 세션 소프트패스 정합성 ---

def test_refresh_clears_soft_pass_anchor_of_deactivated_speaker(db, service):
    service.anchor_session_speaker("dad", "아빠")
    db.profiles = [_record("mom", "엄마")]

    assert service.refresh_speaker_profiles(expected_active_count=1) is True
    assert service._last_passed_user_id is None
    assert service._last_passed_monotonic is None


def test_refresh_updates_soft_pass_display_name_on_rename(db, service):
    service.anchor_session_speaker("dad", "아빠")
    db.profiles = [_record("dad", "아버지")]

    assert service.refresh_speaker_profiles(expected_active_count=1) is True
    assert service._last_passed_user_id == "dad"
    assert service._last_passed_display_name == "아버지"


# --- 스레드 수명 ---

def test_worker_start_records_baseline_and_stop_joins_quickly(db, service, enrollment):
    worker = SpeakerCacheSyncWorker(
        speaker_service_instance=service,
        enrollment_service_instance=enrollment,
        poll_interval_sec=3600,
        fingerprint_loader=db.load_fingerprint,
    )
    worker.start()
    try:
        assert worker._last_fingerprint == (T0, 1, 1)
    finally:
        worker.stop()
    assert not worker._thread.is_alive()
