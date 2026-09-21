"""
tests/test_speaker_identification.py
다중 사용자 1:N 화자 식별(SpeakerService.identify_speaker) 검증 pytest 단위 테스트.

기존 단일 기준 화자 방식(verify(), tests/test_speaker_verification.py)과 달리, DB(speaker_profiles)에
등록된 여러 화자 임베딩 중 코사인 유사도가 가장 높은 후보를 찾는 1:N 매칭을 검증한다.
Resemblyzer 실제 모델은 로딩하지 않고, embed()를 가짜 임베딩으로 대체해 순수 매칭/가드
로직만 검증한다. DB 접근은 server.repositories.speaker_repository.get_active_speaker_embeddings를
모듈 레벨에서 monkeypatch로 대체한다.
"""
from unittest.mock import Mock

import numpy as np
import pytest

import server.services.speaker_service as speaker_service_module
from server.repositories.speaker_repository import SpeakerEmbeddingRecord
from server.services.speaker_service import SpeakerService

# 기준 화자를 2차원 단위벡터로 고정해두면, 후보 벡터를 [cos_sim, sin] 형태로 구성해
# 원하는 코사인 유사도 값을 정확히 재현할 수 있다 (임계값 경계 테스트에 필요).
REFERENCE_A = np.array([1.0, 0.0])
REFERENCE_B = np.array([0.0, 1.0])


def _rotate(base: np.ndarray, cos_sim: float) -> np.ndarray:
    """base 벡터 기준으로 정확히 cos_sim의 코사인 유사도를 갖는 벡터를 만든다 (2D 전용, base가 축 벡터일 때)."""
    orth = np.sqrt(max(0.0, 1.0 - cos_sim**2))
    if np.allclose(base, [1.0, 0.0]):
        return np.array([cos_sim, orth])
    return np.array([orth, cos_sim])


@pytest.fixture()
def service(monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", True)
    return SpeakerService(similarity_threshold=0.65)


def _patch_profiles(monkeypatch, records):
    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", lambda: records)


# --- 스킵 조건 ---

def test_identify_speaker_skips_when_disabled_flag(service, monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", False)
    embed_mock = Mock()
    monkeypatch.setattr(service, "embed", embed_mock)

    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.skipped is True
    assert result.user_id == speaker_service_module.settings.DEFAULT_USER_ID
    embed_mock.assert_not_called()


def test_identify_speaker_skips_when_no_profiles_enrolled(service, monkeypatch):
    _patch_profiles(monkeypatch, [])
    embed_mock = Mock()
    monkeypatch.setattr(service, "embed", embed_mock)

    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.skipped is True
    assert result.user_id == speaker_service_module.settings.DEFAULT_USER_ID
    embed_mock.assert_not_called()


# --- 1:N 매칭 ---

def test_identify_speaker_matches_best_candidate_among_multiple_profiles(service, monkeypatch):
    records = [
        SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist()),
        SpeakerEmbeddingRecord(user_id="mom", display_name="엄마", embedding=REFERENCE_B.tolist()),
    ]
    _patch_profiles(monkeypatch, records)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_B.copy())

    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.user_id == "mom"
    assert result.display_name == "엄마"
    assert result.similarity == pytest.approx(1.0)


def test_identify_speaker_blocks_unregistered_voice_below_threshold(service, monkeypatch):
    records = [
        SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist()),
        SpeakerEmbeddingRecord(user_id="mom", display_name="엄마", embedding=REFERENCE_B.tolist()),
    ]
    _patch_profiles(monkeypatch, records)
    impostor = -REFERENCE_A  # 두 후보 모두와 낮은/음수 유사도
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: impostor)

    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    # impostor([-1,0])는 dad(1.0 유사도 기준 -1.0)보다 mom과의 유사도(0.0)가 더 높지만,
    # 두 후보 모두 임계값(0.65) 미달이라 여전히 미등록 화자로 차단된다.
    assert result.is_match is False
    assert result.user_id is None
    assert result.display_name is None
    assert result.similarity == pytest.approx(0.0)


# --- 짧은 발화 완화 임계값 (verify()와 동일 가드 유지) ---

def test_identify_speaker_short_utterance_uses_relaxed_threshold(service, monkeypatch):
    records = [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist())]
    _patch_profiles(monkeypatch, records)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_MAX_SEC", 1.0)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_THRESHOLD", 0.60)
    candidate = _rotate(REFERENCE_A, 0.65)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: candidate)

    short_audio = np.zeros(8000, dtype=np.float32)  # 16kHz 기준 0.5초
    result = service.identify_speaker(short_audio, sample_rate=16000)

    assert result.is_match is True
    assert result.user_id == "dad"
    assert result.threshold == pytest.approx(0.60)


def test_identify_speaker_long_utterance_uses_default_threshold(service, monkeypatch):
    records = [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist())]
    _patch_profiles(monkeypatch, records)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_MAX_SEC", 1.0)
    candidate = _rotate(REFERENCE_A, 0.60)  # 짧은 발화 완화 임계값(0.60)이면 통과했을 값
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: candidate)

    long_audio = np.zeros(32000, dtype=np.float32)  # 2.0초
    result = service.identify_speaker(long_audio, sample_rate=16000)

    assert result.is_match is False
    assert result.threshold == pytest.approx(0.65)
    assert result.similarity == pytest.approx(0.60)


# --- 세션 소프트패스: 직전 통과 화자로 귀속 ---

def test_identify_speaker_session_soft_pass_retains_previously_passed_user(service, monkeypatch):
    records = [
        SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist()),
        SpeakerEmbeddingRecord(user_id="mom", display_name="엄마", embedding=REFERENCE_B.tolist()),
    ]
    _patch_profiles(monkeypatch, records)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC", 10.0)

    fake_now = {"t": 100.0}
    monkeypatch.setattr(speaker_service_module.time, "monotonic", lambda: fake_now["t"])

    # 1) "mom"이 정상 통과
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_B.copy())
    first = service.identify_speaker(np.zeros(32000, dtype=np.float32))
    assert first.is_match is True
    assert first.user_id == "mom"

    # 2) 3초 후 두 후보 모두와 낮은 유사도의 발화 -> 세션 소프트패스로 직전 화자(mom)에게 귀속
    fake_now["t"] = 103.0
    low_score_candidate = -REFERENCE_A  # dad(-1.0), mom(0.0) 모두 임계값 미달
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: low_score_candidate)
    second = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert second.is_match is True
    assert second.soft_passed is True
    assert second.user_id == "mom"


def test_identify_speaker_without_prior_pass_does_not_soft_pass(service, monkeypatch):
    records = [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist())]
    _patch_profiles(monkeypatch, records)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: -REFERENCE_A)

    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert result.is_match is False
    assert result.soft_passed is False


# --- 예외 시 가용성 우선 폴백 ---

def test_identify_speaker_fails_open_on_embedding_exception(service, monkeypatch):
    records = [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist())]
    _patch_profiles(monkeypatch, records)

    def raise_error(audio, sample_rate=16000):
        raise RuntimeError("resemblyzer backend error")

    monkeypatch.setattr(service, "embed", raise_error)

    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.skipped is True
    assert result.user_id == speaker_service_module.settings.DEFAULT_USER_ID


# --- [실기 회귀] pgvector.Vector 타입 방어 ---
# 실기 테스트 결과, register_vector() 등록 상태/pgvector-python 버전에 따라 speaker_embedding
# 컬럼이 순수 list/np.ndarray가 아닌 pgvector.Vector 객체로 반환되어 코사인 유사도 계산에서
# "unsupported operand type(s) for *: 'Vector' and 'Vector'" TypeError가 발생하고 Fail-Open
# (안전 통과)으로 빠지는 문제가 있었다. cosine_similarity()가 dtype=np.float32 강제 변환으로
# 이를 방어하는지 검증한다.

class FakePgvectorVector:
    """pgvector-python의 Vector 객체를 흉내낸 모의 클래스 (list/ndarray가 아닌 별도 타입)."""

    def __init__(self, values):
        self._values = list(values)

    def to_list(self):
        return self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


def test_cosine_similarity_coerces_list_inputs_to_float_ndarray():
    a = [1.0, 0.0]
    b = [1.0, 0.0]

    assert SpeakerService.cosine_similarity(a, b) == pytest.approx(1.0)


def test_cosine_similarity_coerces_pgvector_like_object_inputs():
    """np.asarray()에 dtype 없이 넘기면 0차원 object 배열로 감싸질 수 있는 커스텀 타입도
    dtype=np.float32 강제 변환 덕분에 정상적으로 원소 단위 float 배열로 처리되어야 한다."""
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = FakePgvectorVector([1.0, 0.0])

    assert SpeakerService.cosine_similarity(a, b) == pytest.approx(1.0)


def test_identify_speaker_handles_pgvector_like_embedding_from_db(service, monkeypatch):
    """SpeakerEmbeddingRecord.embedding이 (정규화 누락 등으로) pgvector.Vector 유사 객체로
    캐싱되어 있어도 identify_speaker()가 TypeError 없이 정상 판정해야 한다."""
    records = [
        SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=FakePgvectorVector(REFERENCE_A.tolist())),
        SpeakerEmbeddingRecord(user_id="mom", display_name="엄마", embedding=FakePgvectorVector(REFERENCE_B.tolist())),
    ]
    _patch_profiles(monkeypatch, records)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_B.copy())

    result = service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert result.skipped is False
    assert result.is_match is True
    assert result.user_id == "mom"
    assert result.similarity == pytest.approx(1.0)


# --- 프로필 캐싱 / reload_speaker_profiles() ---

def test_active_profiles_are_loaded_from_db_only_once(service, monkeypatch):
    records = [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist())]
    load_calls = []

    def spy_load():
        load_calls.append(1)
        return records

    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", spy_load)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_A.copy())

    service.identify_speaker(np.zeros(32000, dtype=np.float32))
    service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert len(load_calls) == 1  # 최초 1회만 DB 조회, 이후는 캐시 사용


def test_reload_speaker_profiles_invalidates_cache(service, monkeypatch):
    load_calls = []

    def spy_load():
        load_calls.append(1)
        return [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist())]

    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", spy_load)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_A.copy())

    service.identify_speaker(np.zeros(32000, dtype=np.float32))
    service.reload_speaker_profiles()
    service.identify_speaker(np.zeros(32000, dtype=np.float32))

    assert len(load_calls) == 2


def test_has_enrolled_speakers_reflects_active_profile_presence(service, monkeypatch):
    _patch_profiles(monkeypatch, [])
    assert service.has_enrolled_speakers() is False

    service.reload_speaker_profiles()
    _patch_profiles(
        monkeypatch, [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE_A.tolist())]
    )
    assert service.has_enrolled_speakers() is True
