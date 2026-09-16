"""
server/services/memory_service.py
RAG 장기 기억 규칙 기반 필터링(B-1) 및 비동기 적재 오케스트레이션.
Gemini 재호출 없이 키워드/길이 휴리스틱만으로 "기억할 가치가 있는 발화"를 판별하여
TTS 완료 후 Background Task에서 임베딩 및 DB 적재를 수행한다.
"""
import logging

from config import settings
from server.repositories.memory_repository import insert_memory
from server.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)

# 개인 정보/취향/일정성 발화를 식별하기 위한 주요 키워드
MEMORABLE_KEYWORDS = [
    "이름은", "생일", "좋아해", "싫어해", "알레르기",
    "매일", "항상", "습관", "취미는", "직업은", "사는 곳",
    "다음 주", "약속", "일정", "기억해",
]

# 키워드가 없어도 충분히 길면(맥락이 담긴 발화로 간주) 기억 후보로 채택
MIN_MEMORABLE_LENGTH = 20


def is_memorable(text: str) -> bool:
    """규칙 기반 필터: 주요 키워드 포함 또는 일정 길이 이상이면 장기 기억 후보로 판단한다."""
    if not text or not text.strip():
        return False

    stripped = text.strip()
    if any(keyword in stripped for keyword in MEMORABLE_KEYWORDS):
        return True

    return len(stripped) >= MIN_MEMORABLE_LENGTH


def extract_and_store(user_text: str, user_id: str = settings.DEFAULT_USER_ID) -> None:
    """
    TTS 완료 후 Background Task(스레드)에서 호출된다.
    규칙 기반 필터를 통과한 발화만 로컬 임베딩 후 pgvector에 원문 그대로 적재한다.
    """
    if not is_memorable(user_text):
        return

    try:
        embedding = embedding_service.embed(user_text)
        insert_memory(user_id=user_id, fact_text=user_text.strip(), embedding=embedding)
    except Exception as e:
        logger.warning(f"[MemoryService] 장기 기억 비동기 적재 실패: {e}")
