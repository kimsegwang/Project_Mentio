"""
tests/test_speech_duration.py
"무음 트리밍 후 실제 발화 길이" 기준 판정 회귀 테스트.

[배경 버그]
캡처 오디오에는 VAD 사전 버퍼(약 0.27초)와 발화 종료 판정용 무음(0.8초 이상)이 항상 포함된다.
기존에는 이 트리밍 전 길이로 판정해서
  A) 짧은 발화 완화 임계값(SPEAKER_SHORT_UTTERANCE_THRESHOLD)이 한 번도 적용되지 않았고
     (0.4초 발화 -> 1.47초로 측정),
  B) 온보딩 최소 샘플 길이(1.5초)를 실제 발화가 0.4초뿐인 샘플도 통과했다.
이 파일은 캡처 길이와 실제 발화 길이가 다른 상황을 재현해 두 판정이 실제 발화 길이를 따르는지 검증한다.
"""
import sys
import types
from unittest.mock import Mock

import numpy as np
import pytest

import server.services.speaker_service as speaker_service_module
import server.services.voice_enrollment_service as enrollment_module
from server.repositories.speaker_repository import SpeakerEmbeddingRecord
from server.schemas.speaker import EnrollmentState
from server.services.speaker_service import SpeakerService
from server.services.voice_enrollment_service import VoiceEnrollmentService

SR = 16000
REFERENCE = np.array([1.0, 0.0])


def _captured(speech_sec: float) -> np.ndarray:
    """리스너 캡처 모양: 사전 버퍼 0.27초 + 발화 + 종료 무음 0.8초 (값은 의미 없음)."""
    return np.zeros(int(SR * (0.27 + speech_sec + 0.8)), dtype=np.float32)


def _trim_to(speech_sec: float):
    """preprocess() 대역: 캡처 길이와 무관하게 speech_sec 길이의 트리밍 결과를 돌려준다."""
    return lambda audio, sample_rate=16000: np.ones(int(SR * speech_sec), dtype=np.float32)


def _at_similarity(cos_sim: float) -> np.ndarray:
    return np.array([cos_sim, np.sqrt(1.0 - cos_sim**2)])


@pytest.fixture()
def service(monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", True)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_MAX_SEC", 1.0)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SHORT_UTTERANCE_THRESHOLD", 0.60)
    monkeypatch.setattr(
        speaker_service_module,
        "get_active_speaker_embeddings",
        lambda: [SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=REFERENCE.tolist())],
    )
    return SpeakerService(similarity_threshold=0.65)


# --- A. 짧은 발화 완화 임계값 ---

def test_short_speech_in_long_capture_uses_relaxed_threshold(service, monkeypatch):
    """캡처는 1.47초지만 실제 발화가 0.4초면 완화 임계값(0.60)으로 판정해야 한다 (기존 버그 재현)."""
    captured = _captured(0.4)
    assert SpeakerService.audio_duration_sec(captured) > 1.0  # 트리밍 전 길이로는 짧은 발화가 아님

    monkeypatch.setattr(service, "preprocess", _trim_to(0.4))
    monkeypatch.setattr(service, "embed_preprocessed", lambda wav: _at_similarity(0.62))

    result = service.identify_speaker(captured)

    assert result.threshold == pytest.approx(0.60)
    assert result.is_match is True
    assert result.user_id == "dad"


def test_long_speech_keeps_default_threshold(service, monkeypatch):
    monkeypatch.setattr(service, "preprocess", _trim_to(2.0))
    monkeypatch.setattr(service, "embed_preprocessed", lambda wav: _at_similarity(0.62))

    result = service.identify_speaker(_captured(2.0))

    assert result.threshold == pytest.approx(0.65)
    assert result.is_match is False


def test_verify_also_uses_speech_duration(service, monkeypatch, tmp_path):
    service.reference_embedding_path = str(tmp_path / "ref.npy")
    np.save(service.reference_embedding_path, REFERENCE)
    monkeypatch.setattr(service, "preprocess", _trim_to(0.4))
    monkeypatch.setattr(service, "embed_preprocessed", lambda wav: _at_similarity(0.62))

    result = service.verify(_captured(0.4))

    assert result.threshold == pytest.approx(0.60)
    assert result.is_match is True


def test_identify_preprocesses_only_once_and_embeds_trimmed_wav(service, monkeypatch):
    """발화 길이 측정과 임베딩이 같은 트리밍 결과를 공유해 VAD 전처리를 중복 수행하지 않는다."""
    trimmed = np.ones(int(SR * 0.4), dtype=np.float32)
    preprocess_mock = Mock(return_value=trimmed)
    embed_mock = Mock(return_value=_at_similarity(0.9))
    monkeypatch.setattr(service, "preprocess", preprocess_mock)
    monkeypatch.setattr(service, "embed_preprocessed", embed_mock)

    service.identify_speaker(_captured(0.4))

    preprocess_mock.assert_called_once()
    assert embed_mock.call_args.args[0] is trimmed


def test_preprocess_exception_falls_back_to_pass(service, monkeypatch):
    """전처리 예외도 기존 임베딩 예외와 동일하게 가용성 우선 폴백(통과, skipped)으로 처리한다."""
    monkeypatch.setattr(service, "preprocess", Mock(side_effect=RuntimeError("webrtcvad error")))

    result = service.identify_speaker(_captured(1.0))

    assert result.is_match is True
    assert result.skipped is True


def test_embed_with_speech_duration_measures_resemblyzer_output(monkeypatch):
    """실제 코드 경로: resemblyzer.preprocess_wav 결과 길이가 곧 발화 길이다."""
    fake_resemblyzer = types.ModuleType("resemblyzer")
    fake_resemblyzer.preprocess_wav = lambda wav, source_sr=None: wav[: int(SR * 0.4)]

    class FakeVoiceEncoder:
        def __init__(self, device="cpu"):
            pass

        def embed_utterance(self, wav):
            return np.full(4, len(wav), dtype=np.float32)

    fake_resemblyzer.VoiceEncoder = FakeVoiceEncoder
    monkeypatch.setitem(sys.modules, "resemblyzer", fake_resemblyzer)

    embedding, speech_sec = SpeakerService().embed_with_speech_duration(_captured(0.4))

    assert speech_sec == pytest.approx(0.4)
    assert embedding[0] == int(SR * 0.4)  # 트리밍된 오디오가 그대로 인코더에 전달됨


def test_real_resemblyzer_preprocess_does_not_count_silence():
    """실제 webrtcvad 트리밍: 무음만 있는 캡처는 발화 길이 0초로 측정된다 (모델 로딩 없음)."""
    pytest.importorskip("webrtcvad")
    pytest.importorskip("resemblyzer")
    service = SpeakerService()

    wav = service.preprocess(_captured(0.0))

    assert SpeakerService.speech_duration_sec(wav) == pytest.approx(0.0)


# --- B. 온보딩 최소 샘플 길이 ---

@pytest.fixture()
def enrollment_parts(monkeypatch):
    upsert = Mock(return_value=True)
    monkeypatch.setattr(enrollment_module, "upsert_speaker_profile", upsert)
    speaker = Mock()
    speaker.speech_duration_sec.side_effect = SpeakerService.speech_duration_sec
    speaker.embed_preprocessed.return_value = np.array([0.0, 1.0], dtype=np.float32)
    speaker.find_closest_profile.return_value = None
    # 샘플 1개의 길이 검사/트리밍 재사용만 검증하므로 다중 샘플 수집은 끈다 (평균화는 test_voice_enrollment.py)
    enrollment = VoiceEnrollmentService(speaker_service_instance=speaker, min_sample_sec=1.5, required_samples=1)
    enrollment.start()
    enrollment.handle_turn(_captured(1.0), "민수")
    assert enrollment.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    speaker.preprocess.reset_mock()  # 이름 발화 임베딩 시도분은 샘플 검증 대상에서 제외
    speaker.embed_preprocessed.reset_mock()
    return enrollment, speaker, upsert


def test_enrollment_rejects_long_capture_with_short_speech(enrollment_parts):
    """캡처는 1.87초(>=1.5초)지만 실제 발화가 0.8초면 부실 샘플로 보고 재요청해야 한다 (기존 버그 재현)."""
    enrollment, speaker, upsert = enrollment_parts
    captured = _captured(0.8)
    assert SpeakerService.audio_duration_sec(captured) >= 1.5
    speaker.preprocess.side_effect = _trim_to(0.8)

    reply = enrollment.handle_turn(captured, "오늘 날씨가 참 좋다")

    assert reply.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    assert reply.completed is False
    speaker.embed_preprocessed.assert_not_called()  # 길이 미달이면 모델 추론 없이 재요청
    upsert.assert_not_called()


def test_enrollment_registers_with_enough_speech_using_trimmed_wav(enrollment_parts):
    enrollment, speaker, upsert = enrollment_parts
    trimmed = np.ones(int(SR * 2.0), dtype=np.float32)
    speaker.preprocess.side_effect = lambda audio, sample_rate=16000: trimmed

    reply = enrollment.handle_turn(_captured(2.0), "오늘 날씨가 참 좋다, 산책 가고 싶어")

    assert reply.completed is True
    speaker.preprocess.assert_called_once()
    assert speaker.embed_preprocessed.call_args.args[0] is trimmed  # 트리밍 결과를 재사용
    upsert.assert_called_once()


def test_enrollment_preprocess_exception_is_handled_as_failure(enrollment_parts):
    enrollment, speaker, upsert = enrollment_parts
    speaker.preprocess.side_effect = RuntimeError("webrtcvad error")

    reply = enrollment.handle_turn(_captured(2.0), "오늘 날씨가 참 좋다")

    assert reply.completed is False
    assert enrollment.state == EnrollmentState.IDLE
    speaker.embed_preprocessed.assert_not_called()
    upsert.assert_not_called()
