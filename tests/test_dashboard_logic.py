"""
tests/test_dashboard_logic.py
관리 대시보드 순수 표시/검증 로직(web/dashboard_logic.py) 단위 테스트 (Streamlit/DB 불필요).
"""
from datetime import datetime, timedelta, timezone

import pytest

from server.schemas.speaker import SpeakerProfile
from web.dashboard_logic import (
    DISPLAY_NAME_MAX_LENGTH,
    build_overview,
    build_user_options,
    format_timestamp,
    validate_display_name,
)


def _speaker(user_id, name, active=True):
    return SpeakerProfile(user_id=user_id, display_name=name, is_active=active)


# --- validate_display_name() ---

def test_validate_display_name_strips_whitespace():
    assert validate_display_name("  아빠 ") == "아빠"


@pytest.mark.parametrize("raw", ["", "   ", None])
def test_validate_display_name_rejects_blank(raw):
    with pytest.raises(ValueError):
        validate_display_name(raw)


def test_validate_display_name_enforces_db_column_length():
    assert validate_display_name("가" * DISPLAY_NAME_MAX_LENGTH) == "가" * DISPLAY_NAME_MAX_LENGTH
    with pytest.raises(ValueError):
        validate_display_name("가" * (DISPLAY_NAME_MAX_LENGTH + 1))


# --- build_overview() ---

def test_build_overview_counts_active_speakers_and_all_memories():
    speakers = [_speaker("dad", "아빠"), _speaker("mom", "엄마", active=False)]

    overview = build_overview(speakers, {"dad": 3, "primary_user": 5})

    assert overview.active_speaker_count == 1
    assert overview.total_speaker_count == 2
    assert overview.total_memory_count == 8


def test_build_overview_handles_empty_db():
    overview = build_overview([], {})
    assert (overview.active_speaker_count, overview.total_speaker_count, overview.total_memory_count) == (0, 0, 0)


# --- build_user_options() ---

def test_build_user_options_orders_active_then_inactive_then_orphans():
    speakers = [_speaker("mom", "엄마", active=False), _speaker("dad", "아빠"), _speaker("kid", "막내")]
    counts = {"dad": 2, "zeta": 1, "primary_user": 4}

    options = build_user_options(speakers, counts)

    assert [o.user_id for o in options] == ["dad", "kid", "mom", "primary_user", "zeta"]
    assert [o.memory_count for o in options] == [2, 0, 0, 4, 1]


def test_build_user_options_labels_reflect_status():
    options = build_user_options([_speaker("mom", "엄마", active=False)], {"primary_user": 4})

    assert "엄마 (mom)" in options[0].label and "비활성" in options[0].label
    assert "화자 미등록" in options[1].label and "기억 4개" in options[1].label


def test_build_user_options_does_not_duplicate_speakers_with_memories():
    options = build_user_options([_speaker("dad", "아빠")], {"dad": 3})
    assert [o.user_id for o in options] == ["dad"]


# --- format_timestamp() ---

def test_format_timestamp_none():
    assert format_timestamp(None) == "-"


def test_format_timestamp_naive():
    assert format_timestamp(datetime(2026, 9, 27, 8, 5)) == "2026-09-27 08:05"


def test_format_timestamp_converts_aware_to_local():
    aware = datetime(2026, 9, 27, 0, 0, tzinfo=timezone(timedelta(hours=9)))
    assert format_timestamp(aware) == aware.astimezone().strftime("%Y-%m-%d %H:%M")
