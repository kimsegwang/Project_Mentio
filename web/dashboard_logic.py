"""
web/dashboard_logic.py
관리 대시보드(web/app.py)의 순수 표시/검증 로직.
Streamlit에 의존하지 않으므로 UI 없이 pytest로 단위 검증할 수 있다.
"""
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from server.schemas.speaker import SpeakerProfile

# speaker_profiles.display_name VARCHAR(100) 제약과 동일
DISPLAY_NAME_MAX_LENGTH = 100


@dataclass(frozen=True)
class UserOption:
    """[탭 2] 기억 조회 대상 선택 드롭다운의 한 항목."""

    user_id: str
    label: str
    memory_count: int


@dataclass(frozen=True)
class DashboardOverview:
    active_speaker_count: int
    total_speaker_count: int
    total_memory_count: int


def validate_display_name(raw: str) -> str:
    """앞뒤 공백을 제거한 표시 이름을 반환한다. 비었거나 DB 컬럼 길이를 넘으면 ValueError."""
    name = (raw or "").strip()
    if not name:
        raise ValueError("이름을 입력해 주세요.")
    if len(name) > DISPLAY_NAME_MAX_LENGTH:
        raise ValueError(f"이름은 {DISPLAY_NAME_MAX_LENGTH}자 이하로 입력해 주세요.")
    return name


def format_timestamp(value: Optional[datetime]) -> str:
    """TIMESTAMPTZ 값을 로컬 시간대 'YYYY-MM-DD HH:MM'으로 표시한다 (None이면 '-')."""
    if value is None:
        return "-"
    if value.tzinfo is not None:
        value = value.astimezone()
    return value.strftime("%Y-%m-%d %H:%M")


def build_overview(speakers: List[SpeakerProfile], memory_counts: Dict[str, int]) -> DashboardOverview:
    return DashboardOverview(
        active_speaker_count=sum(1 for s in speakers if s.is_active),
        total_speaker_count=len(speakers),
        total_memory_count=sum(memory_counts.values()),
    )


def build_user_options(speakers: List[SpeakerProfile], memory_counts: Dict[str, int]) -> List[UserOption]:
    """
    기억 조회 대상 목록을 만든다. 순서: 활성 화자 → 비활성 화자 → 화자 등록 없이 기억만 있는 user_id
    (예: 화자 등록 이전 단일 사용자 시절의 primary_user 기억). 기억이 남아 있는 한 어떤 사용자도 누락하지 않는다.
    """
    options: List[UserOption] = []
    ordered = sorted(speakers, key=lambda s: not s.is_active)  # stable: 입력 순서 유지
    for speaker in ordered:
        count = memory_counts.get(speaker.user_id, 0)
        suffix = "" if speaker.is_active else " · 비활성"
        options.append(
            UserOption(
                user_id=speaker.user_id,
                label=f"{speaker.display_name} ({speaker.user_id}){suffix} — 기억 {count}개",
                memory_count=count,
            )
        )

    speaker_ids = {s.user_id for s in speakers}
    for user_id in sorted(uid for uid in memory_counts if uid not in speaker_ids):
        count = memory_counts[user_id]
        options.append(
            UserOption(user_id=user_id, label=f"{user_id} (화자 미등록) — 기억 {count}개", memory_count=count)
        )
    return options
