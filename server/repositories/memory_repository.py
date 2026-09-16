import logging
from typing import List

from pgvector.psycopg2 import register_vector

from config import settings
from server.repositories.connection import get_db_connection
from server.schemas.memory import EmbeddingVector, MemoryRecord

logger = logging.getLogger(__name__)


def insert_memory(user_id: str, fact_text: str, embedding: EmbeddingVector) -> None:
    """추출된 사실/에피소드 한 건을 장기 기억 저장소에 적재한다 (Background Task 전용)."""
    query = """
        INSERT INTO user_long_term_memory (user_id, fact_text, embedding)
        VALUES (%s, %s, %s::vector);
    """
    try:
        with get_db_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id, fact_text, embedding))
            conn.commit()
            logger.info(f"[MemoryRepository] 장기 기억 적재 완료 (user: {user_id}, text: {fact_text[:30]}...)")
    except Exception as e:
        logger.error(f"[MemoryRepository Error] 장기 기억 적재 실패: {e}")


def search_similar_memories(
    embedding: EmbeddingVector,
    user_id: str = settings.DEFAULT_USER_ID,
    top_k: int = settings.RAG_TOP_K,
) -> List[MemoryRecord]:
    """쿼리 임베딩과 코사인 유사도가 가장 높은 상위 top_k개의 장기 기억을 조회한다."""
    query = """
        SELECT id, user_id, fact_text, created_at, 1 - (embedding <=> %s::vector) AS similarity
        FROM user_long_term_memory
        WHERE user_id = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s;
    """
    try:
        with get_db_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute(query, (embedding, user_id, embedding, top_k))
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
