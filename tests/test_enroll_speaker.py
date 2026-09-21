"""
tests/test_enroll_speaker.py
scripts/enroll_speaker.py CLI 화자 등록 파이프라인 검증 pytest 단위 테스트.

실제 마이크(sounddevice)/DB에 접근하지 않고, 녹음/임베딩/DB 저장 각 단계를 Mock으로
대체해 다음을 검증한다:
- --user-id/--name CLI 인자 파싱 및 기본값(생략 시 DEFAULT_USER_ID 호환)
- enroll()이 녹음 -> 임베딩 추출 -> DB UPSERT -> 캐시 무효화 순서로 호출하는가
- DB 저장 실패 시 캐시 무효화를 호출하지 않고 실패를 보고하는가
"""
import sys
from unittest.mock import Mock

import numpy as np

import scripts.enroll_speaker as enroll_module
from config import settings


def test_parse_args_defaults_to_primary_user_when_omitted(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["enroll_speaker.py"])

    args = enroll_module._parse_args()

    assert args.user_id == settings.DEFAULT_USER_ID
    assert args.name is None
    assert args.duration == enroll_module.RECORD_SECONDS


def test_parse_args_accepts_user_id_and_name(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["enroll_speaker.py", "--user-id", "dad", "--name", "아빠", "--duration", "5"])

    args = enroll_module._parse_args()

    assert args.user_id == "dad"
    assert args.name == "아빠"
    assert args.duration == 5.0


def test_enroll_records_embeds_and_upserts_in_order(monkeypatch):
    call_order = []

    fake_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(
        enroll_module,
        "record_reference_audio",
        lambda duration_sec, sample_rate=16000: (call_order.append("record"), fake_audio)[1],
    )

    fake_embedding = np.ones(256, dtype=np.float32)

    def fake_embed(audio, sample_rate=16000):
        call_order.append("embed")
        return fake_embedding

    monkeypatch.setattr(enroll_module.speaker_service, "embed", fake_embed)

    captured_upsert = {}

    def fake_upsert(user_id, display_name, embedding):
        call_order.append("upsert")
        captured_upsert["user_id"] = user_id
        captured_upsert["display_name"] = display_name
        captured_upsert["embedding"] = embedding
        return True

    monkeypatch.setattr(enroll_module, "upsert_speaker_profile", fake_upsert)

    reload_mock = Mock(side_effect=lambda: call_order.append("reload"))
    monkeypatch.setattr(enroll_module.speaker_service, "reload_speaker_profiles", reload_mock)

    result = enroll_module.enroll(user_id="dad", display_name="아빠", duration_sec=3.0)

    assert result is True
    assert call_order == ["record", "embed", "upsert", "reload"]
    assert captured_upsert["user_id"] == "dad"
    assert captured_upsert["display_name"] == "아빠"
    assert captured_upsert["embedding"] == fake_embedding.tolist()


def test_enroll_does_not_reload_cache_when_db_upsert_fails(monkeypatch):
    fake_audio = np.zeros(16000, dtype=np.float32)
    monkeypatch.setattr(enroll_module, "record_reference_audio", lambda duration_sec, sample_rate=16000: fake_audio)
    monkeypatch.setattr(enroll_module.speaker_service, "embed", lambda audio, sample_rate=16000: np.ones(256))
    monkeypatch.setattr(enroll_module, "upsert_speaker_profile", lambda user_id, display_name, embedding: False)

    reload_mock = Mock()
    monkeypatch.setattr(enroll_module.speaker_service, "reload_speaker_profiles", reload_mock)

    result = enroll_module.enroll(user_id="dad", display_name="아빠")

    assert result is False
    reload_mock.assert_not_called()
