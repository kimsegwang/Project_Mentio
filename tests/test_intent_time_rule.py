"""
tests/test_intent_time_rule.py
IntentService의 2단계 '룰 기반 시간 즉시 처리' 기능을 검증하는 pytest 테스트.

CLAUDE.md 가드레일 검증 대상:
- 5단계 파이프라인 순서(호출어 정규화 -> 룰 기반 시간 -> 시각 부정어 veto -> 시각 요구 -> fallback) 보존
- 시간 질의 감지 시 VOICE_TIME_RULE로 라우팅되어 LLM 호출 없이 로컬 시계로 즉답
"""
from datetime import datetime

import pytest

from server.schemas.action import EmotionType, TriggerType
from server.services.intent_service import IntentService


@pytest.fixture()
def service() -> IntentService:
    return IntentService()


# --- [2단계] 룰 기반 시간 질의 감지 ---

@pytest.mark.parametrize(
    "text",
    [
        "몇 시야?",
        "지금 몇 시야",
        "지금 몇 시",
        "현재 시간",
        "현재 시간 알려줘",
        "시간 좀 알려줘",
        "시간 알려줄래",
        "멘티오야 지금 몇 시야?",
        "멘티오 현재 시간 알려줘",
    ],
)
def test_time_query_routes_to_rule_based_instant(service: IntentService, text: str):
    trigger_type, needs_vision = service.analyze_voice_intent(text)

    assert trigger_type == TriggerType.VOICE_TIME_RULE.value
    assert needs_vision is False


# --- 오탐 방지: '몇 시간'(소요 시간) 등은 시간 질의가 아님 ---

@pytest.mark.parametrize(
    "text",
    [
        "여기서 몇 시간 걸려?",
        "숙제하는데 몇 시간 걸렸어",
        "지금 시간 있어?",
        "지금 시간 돼?",
    ],
)
def test_duration_or_availability_questions_are_not_time_rule(
    service: IntentService, text: str
):
    trigger_type, _ = service.analyze_voice_intent(text)

    assert trigger_type != TriggerType.VOICE_TIME_RULE.value


# --- 시간 규칙 삽입 이후에도 기존 3~5단계 회귀 없는지 확인 ---

@pytest.mark.parametrize(
    "text",
    [
        "보지 마",
        "사진 찍지 마",
        "눈 감아",
    ],
)
def test_vision_veto_still_wins_over_default_chat(service: IntentService, text: str):
    trigger_type, needs_vision = service.analyze_voice_intent(text)

    assert trigger_type == TriggerType.VOICE_CHAT.value
    assert needs_vision is False


@pytest.mark.parametrize(
    "text",
    [
        "봐봐 이거 뭐야?",
        "사진 찍어줘",
        "내 표정 어때?",
    ],
)
def test_explicit_vision_request_still_attaches_snapshot(
    service: IntentService, text: str
):
    trigger_type, needs_vision = service.analyze_voice_intent(text)

    assert trigger_type == TriggerType.VOICE_VISION.value
    assert needs_vision is True


def test_default_fallback_is_pure_text_chat(service: IntentService):
    trigger_type, needs_vision = service.analyze_voice_intent("오늘 하루 어땠어?")

    assert trigger_type == TriggerType.VOICE_CHAT.value
    assert needs_vision is False


def test_empty_text_falls_back_to_voice_chat(service: IntentService):
    trigger_type, needs_vision = service.analyze_voice_intent("   ")

    assert trigger_type == TriggerType.VOICE_CHAT.value
    assert needs_vision is False


# --- build_time_response(): 로컬 시계 기반 즉답 생성 (LLM 미개입) ---

def test_build_time_response_uses_local_clock_am(monkeypatch, service: IntentService):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 17, 9, 5)

    monkeypatch.setattr(
        "server.services.intent_service.datetime", FixedDateTime
    )

    response = service.build_time_response()

    assert response.emotion == EmotionType.HAPPY
    assert response.speech == "지금은 오전 9시 5분이야!"


def test_build_time_response_uses_local_clock_pm_and_noon(
    monkeypatch, service: IntentService
):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 17, 15, 30)

    monkeypatch.setattr(
        "server.services.intent_service.datetime", FixedDateTime
    )

    response = service.build_time_response()

    assert response.emotion == EmotionType.HAPPY
    assert response.speech == "지금은 오후 3시 30분이야!"


def test_build_time_response_midnight_hour_conversion(
    monkeypatch, service: IntentService
):
    class FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            return datetime(2026, 9, 17, 0, 0)

    monkeypatch.setattr(
        "server.services.intent_service.datetime", FixedDateTime
    )

    response = service.build_time_response()

    assert response.speech == "지금은 오전 12시 0분이야!"
