import logging
from typing import List, Optional

from pgvector.psycopg2 import register_vector

from config import settings
from server.repositories.connection import get_db_connection
from server.schemas.memory import EmbeddingVector, MemoryRecord, UserProfileSummary

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


def get_active_memories(
    user_id: str = settings.DEFAULT_USER_ID,
    limit: int = settings.PROFILE_SUMMARY_SOURCE_LIMIT,
) -> List[MemoryRecord]:
    """
    [Memory Summarization] 무효화되지 않은(is_active=TRUE) 기억을 최신순으로 최대 limit개 조회한다.
    MemoryService.summarize_user_profile()이 요약 생성 소스 데이터로 사용한다.
    """
    query = """
        SELECT id, user_id, fact_text, created_at
        FROM user_long_term_memory
        WHERE user_id = %s AND is_active = TRUE
        ORDER BY created_at DESC
        LIMIT %s;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id, limit))
                rows = cursor.fetchall()
                return [
                    MemoryRecord(id=row[0], user_id=row[1], fact_text=row[2], created_at=row[3])
                    for row in rows
                ]
    except Exception as e:
        logger.error(f"[MemoryRepository Error] 활성 기억 목록 조회 실패: {e}")
        return []


def upsert_profile_summary(user_id: str, summary_text: str, source_memory_count: int) -> None:
    """
    [Memory Summarization] 사용자 페르소나 요약을 user_profile_summary에 UPSERT한다.
    MemoryService.summarize_user_profile()에서만 호출된다.
    """
    query = """
        INSERT INTO user_profile_summary (user_id, summary_text, source_memory_count, updated_at)
        VALUES (%s, %s, %s, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id) DO UPDATE
        SET summary_text = EXCLUDED.summary_text,
            source_memory_count = EXCLUDED.source_memory_count,
            updated_at = CURRENT_TIMESTAMP;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id, summary_text, source_memory_count))
            conn.commit()
            logger.info(
                f"[MemoryRepository] 프로필 요약 UPSERT 완료 (user: {user_id}, source_count: {source_memory_count})"
            )
    except Exception as e:
        logger.error(f"[MemoryRepository Error] 프로필 요약 저장 실패: {e}")


def get_profile_summary(user_id: str = settings.DEFAULT_USER_ID) -> Optional[UserProfileSummary]:
    """
    [Memory Summarization] 대화 컨텍스트 조립 시 주입할 사용자 프로필 요약을 조회한다.
    아직 요약이 생성되지 않았거나 조회 실패 시에는 None을 반환해 호출부가 안전하게 생략할 수 있게 한다.
    """
    query = """
        SELECT user_id, summary_text, source_memory_count, updated_at
        FROM user_profile_summary
        WHERE user_id = %s;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id,))
                row = cursor.fetchone()
                if row is None:
                    return None
                return UserProfileSummary(
                    user_id=row[0], summary_text=row[1], source_memory_count=row[2], updated_at=row[3]
                )
    except Exception as e:
        logger.error(f"[MemoryRepository Error] 프로필 요약 조회 실패: {e}")
        return None
