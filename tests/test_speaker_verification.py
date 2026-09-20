"""
tests/test_speaker_verification.py
화자 검증(SpeakerService) 파이프라인 및 AIWorker 통합 훅 검증.

Resemblyzer(torch 기반 화자 임베딩 모델)는 실시간 CPU 추론용 무거운 외부 의존성이므로,
이 테스트는 실제 모델을 로딩하지 않는다. SpeakerService.embed()를 가짜(Fake) 임베딩으로
대체하거나 resemblyzer 모듈 자체를 Fake 모듈로 대역해, 하드웨어/무거운 ML 런타임 없이도
다음을 검증한다:
- 기준 임베딩 미등록 시 검증 스킵(통과)
- 비활성화 플래그(SPEAKER_VERIFICATION_ENABLED=False) 시 검증 스킵(통과)
- 코사인 유사도 임계값 이상/미만 판정
- 임베딩 추출 실패 시 안전하게 통과 처리(가용성 우선, 파이프라인 미차단)
- 등록 파일 캐시(reload_reference_embedding) 동작
- embed()의 PCM16 bytes / float32 배열 입력 처리
- AIWorker 파이프라인이 화자 검증 실패 시 STT 이전 단계에서 조기 반환
"""
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

import server.services.speaker_service as speaker_service_module
import server.workers.ai_worker as ai_worker_module
from server.schemas.action import EmotionType, LLMResponse, TriggerType
from server.services.speaker_service import SpeakerService
from server.workers.ai_worker import AIWorker


def _unit_vector(seed: int, dim: int = 256) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.rand(dim)
    return v / np.linalg.norm(v)


@pytest.fixture()
def service(tmp_path, monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", True)
    return SpeakerService(
        reference_embedding_path=str(tmp_path / "primary_user.npy"),
        similarity_threshold=0.75,
    )


# --- cosine_similarity ---

def test_cosine_similarity_identical_vectors_is_one():
    v = _unit_vector(1)
    assert SpeakerService.cosine_similarity(v, v) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 1.0])
    assert SpeakerService.cosine_similarity(a, b) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector_returns_zero_without_raising():
    a = np.zeros(4)
    b = _unit_vector(2, dim=4)
    assert SpeakerService.cosine_similarity(a, b) == 0.0


# --- 미등록/비활성화 시 스킵 ---

def test_verify_skips_when_reference_not_enrolled(service):
    assert service.is_enrolled() is False
    assert service.verify(np.zeros(16000, dtype=np.float32)) is True


def test_verify_skips_when_disabled_flag(service, monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", False)
    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)

    # 비활성화 시에는 embed()가 아예 호출되지 않아야 한다 (임베딩 추론조차 스킵)
    embed_mock = Mock()
    monkeypatch.setattr(service, "embed", embed_mock)

    assert service.verify(np.zeros(16000, dtype=np.float32)) is True
    embed_mock.assert_not_called()


# --- 코사인 유사도 임계값 판정 ---

def test_verify_passes_when_similarity_above_threshold(service, monkeypatch):
    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: reference)

    assert service.verify(np.zeros(16000, dtype=np.float32)) is True


def test_verify_blocks_when_similarity_below_threshold(service, monkeypatch):
    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)
    impostor = -reference  # 반대 방향 벡터 -> 코사인 유사도 -1.0 (명백히 임계값 미달)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: impostor)

    assert service.verify(np.zeros(16000, dtype=np.float32)) is False


# --- 예외 발생 시 안전하게 통과 처리(가용성 우선) ---

def test_verify_fails_open_when_embedding_extraction_raises(service, monkeypatch):
    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)

    def raise_error(audio, sample_rate=16000):
        raise RuntimeError("resemblyzer backend error")

    monkeypatch.setattr(service, "embed", raise_error)

    assert service.verify(np.zeros(16000, dtype=np.float32)) is True


# --- 기준 임베딩 캐시 / reload_reference_embedding() ---

def test_reference_embedding_is_loaded_from_disk_only_once(service, monkeypatch):
    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: reference)

    load_calls = []
    original_load = np.load

    def spy_load(path, *args, **kwargs):
        load_calls.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(speaker_service_module.np, "load", spy_load)

    service.verify(np.zeros(16000, dtype=np.float32))
    service.verify(np.zeros(16000, dtype=np.float32))

    assert len(load_calls) == 1  # 최초 1회만 디스크에서 로딩, 이후는 캐시 사용


def test_reload_reference_embedding_picks_up_newly_enrolled_file(service):
    assert service.is_enrolled() is False

    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)

    # reload 전에는 "미등록"으로 캐시된 상태가 그대로 유지된다
    assert service.is_enrolled() is False

    service.reload_reference_embedding()
    assert service.is_enrolled() is True


# --- embed(): resemblyzer를 Fake 모듈로 대역해 bytes/array 입력 처리만 검증 ---

def test_embed_accepts_pcm16_bytes_and_delegates_to_resemblyzer(monkeypatch, service):
    captured = {}

    fake_resemblyzer = types.ModuleType("resemblyzer")

    def fake_preprocess_wav(wav, source_sr=None):
        captured["wav"] = wav
        captured["source_sr"] = source_sr
        return wav

    class FakeVoiceEncoder:
        def __init__(self, device):
            captured["device"] = device

        def embed_utterance(self, wav):
            return np.ones(4, dtype=np.float32)

    fake_resemblyzer.preprocess_wav = fake_preprocess_wav
    fake_resemblyzer.VoiceEncoder = FakeVoiceEncoder
    monkeypatch.setitem(sys.modules, "resemblyzer", fake_resemblyzer)

    pcm16_bytes = np.array([0, 16384, -16384], dtype=np.int16).tobytes()
    result = service.embed(pcm16_bytes, sample_rate=16000)

    assert isinstance(result, np.ndarray)
    assert captured["source_sr"] == 16000
    assert captured["device"] == "cpu"
    np.testing.assert_allclose(
        captured["wav"], np.array([0.0, 0.5, -0.5], dtype=np.float32), atol=1e-4
    )


def test_embed_accepts_float32_array_without_bytes_conversion(monkeypatch, service):
    captured = {}

    fake_resemblyzer = types.ModuleType("resemblyzer")

    def fake_preprocess_wav(wav, source_sr=None):
        captured["wav"] = wav
        return wav

    class FakeVoiceEncoder:
        def __init__(self, device):
            pass

        def embed_utterance(self, wav):
            return np.ones(4, dtype=np.float32)

    fake_resemblyzer.preprocess_wav = fake_preprocess_wav
    fake_resemblyzer.VoiceEncoder = FakeVoiceEncoder
    monkeypatch.setitem(sys.modules, "resemblyzer", fake_resemblyzer)

    audio_array = np.zeros(16000, dtype=np.float32)
    service.embed(audio_array, sample_rate=16000)

    assert captured["wav"] is audio_array


# --- AIWorker 통합: 화자 검증 실패 시 STT 이전 단계에서 조기 반환 ---

def _build_worker(speaker_service_instance):
    mock_brain = Mock()
    mock_brain.infer_action.return_value = LLMResponse(emotion=EmotionType.HAPPY, speech="테스트 응답")
    mock_tts = Mock()
    mock_tts.synthesize.return_value = b""
    mock_audio_player = Mock()
    return AIWorker(
        brain_service_instance=mock_brain,
        tts_service_instance=mock_tts,
        audio_player_instance=mock_audio_player,
        speaker_service_instance=speaker_service_instance,
    )


def test_process_voice_interaction_blocks_before_stt_when_speaker_rejected(monkeypatch):
    rejecting_speaker_service = Mock()
    rejecting_speaker_service.verify.return_value = False
    worker = _build_worker(rejecting_speaker_service)

    stt_mock = Mock()
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", stt_mock)

    audio_data = np.zeros(16000, dtype=np.float32)
    result = worker.process_voice_interaction(audio_data=audio_data)

    assert result is None
    stt_mock.assert_not_called()
    rejecting_speaker_service.verify.assert_called_once_with(audio_data)


def test_process_voice_interaction_continues_when_speaker_accepted(monkeypatch):
    accepting_speaker_service = Mock()
    accepting_speaker_service.verify.return_value = True
    worker = _build_worker(accepting_speaker_service)

    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "안녕")
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "analyze_voice_intent",
        lambda text: (TriggerType.VOICE_CHAT.value, False),
    )
    monkeypatch.setattr(worker, "_retrieve_memory_context", lambda user_text: "")
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", Mock())
    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", Mock())

    action = worker.process_voice_interaction(audio_data=np.zeros(16000, dtype=np.float32))

    assert action is not None
    assert action.speech == "테스트 응답"
    accepting_speaker_service.verify.assert_called_once()


def test_default_ai_worker_uses_global_speaker_service_singleton():
    """DI 기본값이 전역 speaker_service 싱글톤을 가리켜, 실제 조립 지점(main.py)에서
    별도 인자 없이 생성해도 화자 검증이 자연스럽게 연결되어야 한다."""
    worker = AIWorker(
        brain_service_instance=Mock(),
        tts_service_instance=Mock(),
        audio_player_instance=Mock(),
    )
    assert worker.speaker_service is speaker_service_module.speaker_service
