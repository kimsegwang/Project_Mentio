import logging
from typing import List, Optional

from pgvector.psycopg2 import register_vector

from config import settings
from server.repositories.connection import get_db_connection
from server.schemas.memory import EmbeddingVector, MemoryRecord

logger = logging.getLogger(__name__)


def insert_memory(user_id: str, fact_text: str, embedding: EmbeddingVector) -> Optional[int]:
    """
    추출된 사실/에피소드 한 건을 장기 기억 저장소에 적재한다 (Background Task 전용).
    모순 판정 시 신규 행을 superseded_by로 가리키게 하기 위해 신규 행의 id를 반환한다.
    """
    query = """
        INSERT INTO user_long_term_memory (user_id, fact_text, embedding)
        VALUES (%s, %s, %s::vector)
        RETURNING id;
    """
    try:
        with get_db_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id, fact_text, embedding))
                new_id = cursor.fetchone()[0]
            conn.commit()
            logger.info(f"[MemoryRepository] 장기 기억 적재 완료 (user: {user_id}, id: {new_id}, text: {fact_text[:30]}...)")
            return new_id
    except Exception as e:
        logger.error(f"[MemoryRepository Error] 장기 기억 적재 실패: {e}")
        return None


def invalidate_memories(ids: List[int], superseded_by: Optional[int] = None) -> None:
    """
    모순(Invalidation)으로 판정된 기존 기억 행들을 물리적으로 DELETE하지 않고
    is_active=FALSE로만 비활성화한다 (장기 기억 유실 방지, 추후 이력 조회/복구 여지 보존).
    """
    if not ids:
        return

    query = """
        UPDATE user_long_term_memory
        SET is_active = FALSE, superseded_by = %s, updated_at = CURRENT_TIMESTAMP
        WHERE id = ANY(%s);
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (superseded_by, ids))
            conn.commit()
            logger.info(f"[MemoryRepository] 기존 기억 {ids}건 비활성화 완료 (superseded_by: {superseded_by})")
    except Exception as e:
        logger.error(f"[MemoryRepository Error] 기억 비활성화 실패: {e}")


def search_similar_memories(
    embedding: EmbeddingVector,
    user_id: str = settings.DEFAULT_USER_ID,
    top_k: int = settings.RAG_TOP_K,
    threshold: float = settings.RAG_SIMILARITY_THRESHOLD,
) -> List[MemoryRecord]:
    """
    쿼리 임베딩과 코사인 유사도가 threshold 이상인 기억 중 상위 top_k개를 조회한다.
    무효화(is_active=FALSE)된 기억은 Dedup 검사/RAG 주입 어느 쪽에도 노출되면 안 되므로 제외한다.
    """
    query = """
        SELECT id, user_id, fact_text, created_at, 1 - (embedding <=> %s::vector) AS similarity
        FROM user_long_term_memory
        WHERE user_id = %s
          AND is_active = TRUE
          AND 1 - (embedding <=> %s::vector) >= %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    try:
        with get_db_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute(query, (embedding, user_id, embedding, threshold, embedding, top_k))
                rows = cursor.fetchall()
                return [
                    MemoryRecord(
                        id=row[0],
                        user_id=row[1],
                        fact_text=row[2],
                        created_at=row[3],
                        similarity=float(row[4]),
                    )
                    for row in rows
                ]
    except Exception as e:
        logger.error(f"[MemoryRepository Error] 장기 기억 검색 실패: {e}")
        return []
