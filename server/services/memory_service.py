"""
server/services/memory_service.py
RAG 장기 기억 규칙 기반 필터링(B-1) 및 비동기 적재 오케스트레이션.
Gemini 재호출 없이 키워드/길이 휴리스틱만으로 "기억할 가치가 있는 발화"를 판별하여
TTS 완료 후 Background Task에서 임베딩 및 DB 적재를 수행한다.
"""
import logging

from config import settings
from server.repositories.memory_repository import insert_memory, invalidate_memories, search_similar_memories
from server.schemas.memory import MemoryRelation
from server.services.brain_service import brain_service
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
    TTS 완료 후 Background Task(MemoryWriteWorker의 순차 스레드)에서 호출된다.
    규칙 기반 필터를 통과한 발화만 로컬 임베딩 후:
      1) 근접 중복(RAG_DEDUP_SIMILARITY_THRESHOLD 이상)이면 저장 스킵
      2) 주제가 겹치는 후보(RAG_CONFLICT_CANDIDATE_THRESHOLD 이상 ~ Dedup 임계값 미만)가 있으면
         경량 LLM Reflection(BrainService.classify_memory_relation)으로 모순 여부 판정 후,
         모순이면 기존 기억을 is_active=FALSE로 무효화하고 신규 사실로 대체
      3) 그 외에는 그대로 신규 적재
    """
    if not is_memorable(user_text):
        return

    try:
        embedding = embedding_service.embed(user_text)
        fact_text = user_text.strip()

        # 임계값 튜닝 가시성 + 모순 후보 확보를 한 번의 조회로 처리한다
        # (threshold=0.0, top_k=RAG_CONFLICT_TOP_K로 가장 가까운 기존 기억들을 무조건 조회).
        nearest = search_similar_memories(embedding, user_id=user_id, top_k=settings.RAG_CONFLICT_TOP_K, threshold=0.0)
        top_match = nearest[0] if nearest else None

        if top_match and top_match.similarity >= settings.RAG_DEDUP_SIMILARITY_THRESHOLD:
            logger.info(
                f"🧠 [MemoryService] 중복 기억 감지(유사도: {top_match.similarity:.2f}), 저장 스킵: {user_text[:30]}..."
            )
            return

        conflict_candidates = [
            m for m in nearest
            if settings.RAG_CONFLICT_CANDIDATE_THRESHOLD <= (m.similarity or 0.0) < settings.RAG_DEDUP_SIMILARITY_THRESHOLD
        ]

        if conflict_candidates:
            relation_result = brain_service.classify_memory_relation(fact_text, conflict_candidates)
        else:
            relation_result = None

        new_id = insert_memory(user_id=user_id, fact_text=fact_text, embedding=embedding)

        if relation_result is not None and relation_result.relation == MemoryRelation.CONTRADICTS:
            # LLM이 후보 목록 밖의 id를 언급(환각)하는 경우를 방어하기 위해 후보 id로 한 번 더 검증한다.
            candidate_ids = {m.id for m in conflict_candidates}
            ids_to_invalidate = [i for i in relation_result.conflicting_ids if i in candidate_ids]
            if ids_to_invalidate:
                invalidate_memories(ids_to_invalidate, superseded_by=new_id)
                logger.info(
                    f"🧠 [MemoryService] 모순 감지, 기존 기억 {ids_to_invalidate}건 비활성화 "
                    f"-> 신규 기억(id={new_id})으로 대체: {user_text[:30]}..."
                )

        if top_match:
            logger.info(
                f"🧠 [MemoryService] 신규 기억 적재 (최고 유사도: {top_match.similarity:.2f} < "
                f"{settings.RAG_DEDUP_SIMILARITY_THRESHOLD}): {user_text[:30]}..."
            )
        else:
            logger.info(f"🧠 [MemoryService] 신규 기억 적재: {user_text[:30]}...")
    except Exception as e:
        logger.warning(f"[MemoryService] 장기 기억 비동기 적재 실패: {e}")
