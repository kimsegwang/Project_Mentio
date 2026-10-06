"""
server/services/time_service.py
로컬 시계(datetime) 기반 시간/날짜 문자열 포맷 헬퍼.
룰 기반 즉답(IntentService.build_time_response)과 LLM 시스템 프롬프트 일시 주입
(BrainService._build_system_instruction)이 같은 한국어 요일/포맷 규칙을 공유하도록 분리한다.
datetime.now() 호출은 각 호출부가 담당하고, 이 모듈은 전달받은 시각을 순수하게 포맷만 한다
(테스트에서 호출부 모듈의 datetime만 고정하면 되도록).
"""
from datetime import datetime

KOREAN_WEEKDAYS = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일")

TIMEZONE_LABEL = "서울/대한민국 기준"


def korean_weekday(now: datetime) -> str:
    return KOREAN_WEEKDAYS[now.weekday()]


def format_time_phrase(now: datetime) -> str:
    """'오후 8시 24분' 형태의 12시간제 구어체 시각."""
    period = "오전" if now.hour < 12 else "오후"
    hour_12 = now.hour % 12
    hour_12 = 12 if hour_12 == 0 else hour_12
    return f"{period} {hour_12}시 {now.minute}분"


def format_date_phrase(now: datetime) -> str:
    """'2026년 10월 6일 화요일' 형태의 구어체 날짜."""
    return f"{now.year}년 {now.month}월 {now.day}일 {korean_weekday(now)}"


def format_prompt_timestamp(now: datetime) -> str:
    """LLM 시스템 프롬프트 헤더용 '[현재 시각: 2026-10-06 20:24 (화요일), 서울/대한민국 기준]'."""
    return f"[현재 시각: {now.strftime('%Y-%m-%d %H:%M')} ({korean_weekday(now)}), {TIMEZONE_LABEL}]"
