"""
tests/test_speaker_verification.py
화자 검증(SpeakerService) 파이프라인 및 AIWorker 통합 훅 검증.

Resemblyzer(torch 기반 화자 임베딩 모델)는 실시간 CPU 추론용 무거운 외부 의존성이므로,
이 테스트는 실제 모델을 로딩하지 않는다. SpeakerService.embed()를 가짜(Fake) 임베딩으로
대체하거나 resemblyzer 모듈 자체를 Fake 모듈로 대역해, 하드웨어/무거운 ML 런타임 없이도
다음을 검증한다:
- 기준 임베딩 미등록/비활성화 플래그 시 검증 스킵(통과, skipped=True)
- 코사인 유사도 임계값 이상/미만 판정 및 실측 점수(similarity)/적용 임계값(threshold) 노출
- 임베딩 추출 실패 시 안전하게 통과 처리(가용성 우선, 파이프라인 미차단)
- 등록 파일 캐시(reload_reference_embedding) 동작
- embed()의 PCM16 bytes / float32 배열 입력 처리
- [튜닝] 짧은 발화(<= SPEAKER_SHORT_UTTERANCE_MAX_SEC)에 완화된 임계값 적용
- [튜닝] 직전 통과로부터 SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC 이내면 세션 소프트패스로 통과
- AIWorker 파이프라인이 화자 검증 실패 시 STT 이전 단계에서 조기 반환하고, 통과/차단 시
  실측 유사도 점수를 콘솔에 출력하는지 검증
"""
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

import server.services.speaker_service as speaker_service_module
import server.workers.ai_worker as ai_worker_module
from server.schemas.action import EmotionType, LLMResponse, TriggerType
from server.schemas.speaker import SpeakerIdentificationResult, SpeakerVerificationResult
from server.services.speaker_service import SpeakerService
from server.workers.ai_worker import AIWorker

# 기준 화자를 2차원 단위벡터 [1, 0]으로 고정해두면, 후보 벡터를 [cos_sim, sin] 형태로
# 구성해 원하는 코사인 유사도 값을 정확히 재현할 수 있다 (임계값 경계 테스트에 필요).
REFERENCE_2D = np.array([1.0, 0.0])


def _candidate_at_similarity(cos_sim: float) -> np.ndarray:
    """REFERENCE_2D 기준으로 정확히 cos_sim의 코사인 유사도를 갖는 단위벡터를 만든다."""
    orth = np.sqrt(max(0.0, 1.0 - cos_sim**2))
    return np.array([cos_sim, orth])


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

    result = service.verify(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.skipped is True
    assert result.similarity is None


def test_verify_skips_when_disabled_flag(service, monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", False)
    np.save(service.reference_embedding_path, REFERENCE_2D)

    # 비활성화 시에는 embed()가 아예 호출되지 않아야 한다 (임베딩 추론조차 스킵)
    embed_mock = Mock()
    monkeypatch.setattr(service, "embed", embed_mock)

    result = service.verify(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.skipped is True
    embed_mock.assert_not_called()


# --- 코사인 유사도 임계값 판정 (2.0초, 짧은 발화 기준 밖) ---

def test_verify_passes_when_similarity_above_threshold(service, monkeypatch):
    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: reference)

    result = service.verify(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.similarity == pytest.approx(1.0)


def test_verify_blocks_when_similarity_below_threshold(service, monkeypatch):
    reference = _unit_vector(1)
    np.save(service.reference_embedding_path, reference)
    impostor = -reference  # 반대 방향 벡터 -> 코사인 유사도 -1.0 (명백히 임계값 미달)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: impostor)

    result = service.verify(np.zeros(32000, dtype=np.float32))

    assert result.is_match is False
    assert result.similarity == pytest.approx(-1.0)
    assert result.threshold == pytest.approx(0.75)


# --- 예외 발생 시 안전하게 통과 처리(가용성 우선) ---

def test_verify_fails_open_when_embedding_extraction_raises(service, monkeypatch):
    np.save(service.reference_embedding_path, REFERENCE_2D)

    def raise_error(audio, sample_rate=16000):
        raise RuntimeError("resemblyzer backend error")

    monkeypatch.setattr(service, "embed", raise_error)

    result = service.verify(np.zeros(32000, dtype=np.float32))

    assert result.is_match is True
    assert result.skipped is True


# --- 기준 임베딩 캐시 / reload_reference_embedding() ---

def test_reference_embedding_is_loaded_from_disk_only_once(service, monkeypatch):
    np.save(service.reference_embedding_path, REFERENCE_2D)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_2D.copy())

    load_calls = []
    original_load = np.load

    def spy_load(path, *args, **kwargs):
        load_calls.append(path)
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(speaker_service_module.np, "load", spy_load)

    service.verify(np.zeros(32000, dtype=np.float32))
    service.verify(np.zeros(32000, dtype=np.float32))

    assert len(load_calls) == 1  # 최초 1회만 디스크에서 로딩, 이후는 캐시 사용


def test_reload_reference_embedding_picks_up_newly_enrolled_file(service):
    assert service.is_enrolled() is False

    np.save(service.reference_embedding_path, REFERENCE_2D)

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


# --- [튜닝] 짧은 발화 완화 임계값 ---

def test_short_utterance_uses_relaxed_threshold_instead_of_default(service, monkeypatch):
    """1초 이하 발화는 기본 임계값(0.75)엔 미달해도 완화 임계값(0.60)이면 통과해야 한다."""
    np.save(service.reference_embedding_path, REFERENCE_2D)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_MAX_SEC", 1.0)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_THRESHOLD", 0.60)
    candidate = _candidate_at_similarity(0.65)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: candidate)

    short_audio = np.zeros(8000, dtype=np.float32)  # 16kHz 기준 0.5초 (짧은 발화)
    result = service.verify(short_audio, sample_rate=16000)

    assert result.is_match is True
    assert result.threshold == pytest.approx(0.60)
    assert result.similarity == pytest.approx(0.65, abs=1e-6)
    assert result.soft_passed is False


def test_long_utterance_still_uses_default_threshold(service, monkeypatch):
    """같은 유사도(0.65)라도 1초를 넘는 발화는 완화 임계값이 아닌 기본 임계값(0.75)으로 판정한다."""
    np.save(service.reference_embedding_path, REFERENCE_2D)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_MAX_SEC", 1.0)
    candidate = _candidate_at_similarity(0.65)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: candidate)

    long_audio = np.zeros(32000, dtype=np.float32)  # 16kHz 기준 2.0초 (짧은 발화 아님)
    result = service.verify(long_audio, sample_rate=16000)

    assert result.is_match is False
    assert result.threshold == pytest.approx(0.75)


def test_utterance_at_exact_boundary_is_treated_as_short(service, monkeypatch):
    """정확히 SPEAKER_SHORT_UTTERANCE_MAX_SEC와 같은 길이는 짧은 발화로 취급한다(<=)."""
    np.save(service.reference_embedding_path, REFERENCE_2D)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_MAX_SEC", 1.0)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_THRESHOLD", 0.60)
    candidate = _candidate_at_similarity(0.65)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: candidate)

    exact_1s_audio = np.zeros(16000, dtype=np.float32)  # 16kHz 기준 정확히 1.0초
    result = service.verify(exact_1s_audio, sample_rate=16000)

    assert result.threshold == pytest.approx(0.60)
    assert result.is_match is True


# --- [튜닝] 대화 세션 소프트패스 ---

def test_session_soft_pass_allows_low_score_shortly_after_a_real_pass(service, monkeypatch):
    np.save(service.reference_embedding_path, REFERENCE_2D)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC", 10.0)

    fake_now = {"t": 100.0}
    monkeypatch.setattr(speaker_service_module.time, "monotonic", lambda: fake_now["t"])

    # 1) 정상 통과 (임계값 이상)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_2D.copy())
    first = service.verify(np.zeros(32000, dtype=np.float32))
    assert first.is_match is True
    assert first.soft_passed is False

    # 2) 3초 후 임계값 미달 발화 -> 최근 통과 이력 덕분에 세션 소프트패스로 통과
    fake_now["t"] = 103.0
    low_score_candidate = _candidate_at_similarity(0.30)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: low_score_candidate)
    second = service.verify(np.zeros(32000, dtype=np.float32))

    assert second.is_match is True
    assert second.soft_passed is True
    assert second.similarity == pytest.approx(0.30, abs=1e-6)


def test_session_soft_pass_expires_after_window(service, monkeypatch):
    np.save(service.reference_embedding_path, REFERENCE_2D)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC", 10.0)

    fake_now = {"t": 200.0}
    monkeypatch.setattr(speaker_service_module.time, "monotonic", lambda: fake_now["t"])

    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: REFERENCE_2D.copy())
    service.verify(np.zeros(32000, dtype=np.float32))

    fake_now["t"] = 215.0  # 15초 경과 -> 세션 윈도우 만료
    low_score_candidate = _candidate_at_similarity(0.30)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: low_score_candidate)
    result = service.verify(np.zeros(32000, dtype=np.float32))

    assert result.is_match is False
    assert result.soft_passed is False


def test_verify_without_prior_pass_does_not_soft_pass(service, monkeypatch):
    np.save(service.reference_embedding_path, REFERENCE_2D)
    low_score_candidate = _candidate_at_similarity(0.30)
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: low_score_candidate)

    result = service.verify(np.zeros(32000, dtype=np.float32))

    assert result.is_match is False
    assert result.soft_passed is False


# --- 설정값 회귀 방지: 기본 임계값 0.65 최종 조정 ---

def test_default_verification_threshold_lowered_to_0_65():
    import config.settings as raw_settings

    assert raw_settings.SPEAKER_VERIFICATION_THRESHOLD == pytest.approx(0.65)


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
    rejecting_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=False, similarity=0.412, threshold=0.68
    )
    worker = _build_worker(rejecting_speaker_service)

    stt_mock = Mock()
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", stt_mock)

    audio_data = np.zeros(32000, dtype=np.float32)
    result = worker.process_voice_interaction(audio_data=audio_data)

    assert result is None
    stt_mock.assert_not_called()
    rejecting_speaker_service.identify_speaker.assert_called_once_with(audio_data)


def test_process_voice_interaction_continues_when_speaker_accepted(monkeypatch):
    accepting_speaker_service = Mock()
    accepting_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=True, similarity=0.912, threshold=0.68, user_id="primary_user"
    )
    worker = _build_worker(accepting_speaker_service)

    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "안녕")
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "analyze_voice_intent",
        lambda text: (TriggerType.VOICE_CHAT.value, False),
    )
    monkeypatch.setattr(worker, "_retrieve_memory_context", lambda user_text, user_id: "")
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", Mock())
    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", Mock())

    action = worker.process_voice_interaction(audio_data=np.zeros(32000, dtype=np.float32))

    assert action is not None
    assert action.speech == "테스트 응답"
    accepting_speaker_service.identify_speaker.assert_called_once()


def test_process_voice_interaction_prints_actual_similarity_score_on_block(monkeypatch, capsys):
    rejecting_speaker_service = Mock()
    rejecting_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=False, similarity=0.412, threshold=0.68
    )
    worker = _build_worker(rejecting_speaker_service)

    worker.process_voice_interaction(audio_data=np.zeros(32000, dtype=np.float32))

    captured = capsys.readouterr()
    assert "화자 식별 점수" in captured.out
    assert "0.412" in captured.out
    assert "0.68" in captured.out


def test_process_voice_interaction_prints_actual_similarity_score_on_pass(monkeypatch, capsys):
    accepting_speaker_service = Mock()
    accepting_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=True, similarity=0.912, threshold=0.68, user_id="primary_user"
    )
    worker = _build_worker(accepting_speaker_service)

    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "안녕")
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "analyze_voice_intent",
        lambda text: (TriggerType.VOICE_CHAT.value, False),
    )
    monkeypatch.setattr(worker, "_retrieve_memory_context", lambda user_text, user_id: "")
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", Mock())
    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", Mock())

    worker.process_voice_interaction(audio_data=np.zeros(32000, dtype=np.float32))

    captured = capsys.readouterr()
    assert "화자 식별 점수" in captured.out
    assert "0.912" in captured.out


def test_process_voice_interaction_skips_score_print_when_verification_skipped(monkeypatch, capsys):
    """미등록/비활성화로 skipped=True인 경우 similarity가 없어 점수 출력 없이 조용히 통과해야 한다."""
    skipping_speaker_service = Mock()
    skipping_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=True, skipped=True, user_id="primary_user"
    )
    worker = _build_worker(skipping_speaker_service)

    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "안녕")
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "analyze_voice_intent",
        lambda text: (TriggerType.VOICE_CHAT.value, False),
    )
    monkeypatch.setattr(worker, "_retrieve_memory_context", lambda user_text, user_id: "")
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", Mock())
    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", Mock())

    action = worker.process_voice_interaction(audio_data=np.zeros(32000, dtype=np.float32))

    assert action is not None
    captured = capsys.readouterr()
    assert "화자 식별 점수" not in captured.out


def test_default_ai_worker_uses_global_speaker_service_singleton():
    """DI 기본값이 전역 speaker_service 싱글톤을 가리켜, 실제 조립 지점(main.py)에서
    별도 인자 없이 생성해도 화자 검증이 자연스럽게 연결되어야 한다."""
    worker = AIWorker(
        brain_service_instance=Mock(),
        tts_service_instance=Mock(),
        audio_player_instance=Mock(),
    )
    assert worker.speaker_service is speaker_service_module.speaker_service
