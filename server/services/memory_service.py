"""
server/services/memory_service.py
RAG 장기 기억 규칙 기반 필터링(B-1) 및 비동기 적재 오케스트레이션.
Gemini 재호출 없이 키워드/길이 휴리스틱만으로 "기억할 가치가 있는 발화"를 판별하여
TTS 완료 후 Background Task에서 임베딩 및 DB 적재를 수행한다.
"""
import logging

from config import settings
from server.repositories.memory_repository import insert_memory, search_similar_memories
from server.services.embedding_service import embedding_service

logger = logging.getLogger(__name__)
if not logger.handlers:
    # 루트 로거에 핸들러/레벨이 구성되어 있지 않으면(기본 WARNING) INFO 로그가
    # 콘솔에 전혀 출력되지 않으므로, 중복 판별 로그의 실시간 가시성 확보를 위해
    # 이 모듈 전용 콘솔 핸들러를 직접 구성한다.
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

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

        # 임계값 튜닝 가시성을 위해 threshold=0.0으로 가장 가까운 기존 기억 1건을 무조건 조회한다
        # (중복 여부 판정은 아래에서 RAG_DEDUP_SIMILARITY_THRESHOLD와 직접 비교).
        nearest = search_similar_memories(embedding, user_id=user_id, top_k=1, threshold=0.0)

        if nearest and nearest[0].similarity >= settings.RAG_DEDUP_SIMILARITY_THRESHOLD:
            logger.info(
                f"🧠 [MemoryService] 중복 기억 감지(유사도: {nearest[0].similarity:.2f}), 저장 스킵: {user_text[:30]}..."
            )
            return

        insert_memory(user_id=user_id, fact_text=user_text.strip(), embedding=embedding)

        if nearest:
            logger.info(
                f"🧠 [MemoryService] 신규 기억 적재 (최고 유사도: {nearest[0].similarity:.2f} < "
                f"{settings.RAG_DEDUP_SIMILARITY_THRESHOLD}): {user_text[:30]}..."
            )
        else:
            logger.info(f"🧠 [MemoryService] 신규 기억 적재: {user_text[:30]}...")
    except Exception as e:
        logger.warning(f"[MemoryService] 장기 기억 비동기 적재 실패: {e}")
