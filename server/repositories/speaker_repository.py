"""
server/repositories/speaker_repository.py
다중 사용자 화자 프로필(speaker_profiles) DB 접근 전담.
AI 추론/코사인 유사도 계산 등 도메인 로직은 포함하지 않고, 순수 조회/적재만 담당한다
(1:N 유사도 비교 자체는 server/services/speaker_service.py에서 수행).
"""
import logging
from typing import List, Optional

from pgvector.psycopg2 import register_vector

from server.repositories.connection import get_db_connection
from server.schemas.speaker import SpeakerEmbeddingVector, SpeakerProfile

logger = logging.getLogger(__name__)


def _to_embedding_list(value) -> SpeakerEmbeddingVector:
    """
    DB에서 조회한 vector 컬럼 값을 순수 float list로 정규화한다.

    [pgvector 타입 방어] register_vector() 등록 여부/pgvector-python 버전에 따라 조회 결과가
    np.ndarray가 아닌 pgvector.Vector 객체로 반환될 수 있다. Vector끼리 numpy 연산(np.dot 등)에
    바로 사용하면 "unsupported operand type(s) for *: 'Vector' and 'Vector'"처럼 타입 에러가
    발생하므로, 리포지토리 경계에서 즉시 list[float]로 변환해 상위 계층(SpeakerService)이
    드라이버별 반환 타입을 신경 쓰지 않도록 한다.
    """
    if hasattr(value, "to_list"):
        return value.to_list()
    if hasattr(value, "tolist"):
        return value.tolist()
    return list(value)


class SpeakerEmbeddingRecord:
    """1:N 유사도 비교용으로 캐싱되는 (user_id, display_name, embedding) 묶음."""

    __slots__ = ("user_id", "display_name", "embedding")

    def __init__(self, user_id: str, display_name: str, embedding: SpeakerEmbeddingVector):
        self.user_id = user_id
        self.display_name = display_name
        self.embedding = embedding


def upsert_speaker_profile(user_id: str, display_name: str, embedding: SpeakerEmbeddingVector) -> bool:
    """
    화자 프로필을 신규 등록하거나(최초) 임베딩/이름을 갱신한다(재등록).
    scripts/enroll_speaker.py 전용 쓰기 경로이며, 성공 여부를 bool로 반환한다.
    """
    query = """
        INSERT INTO speaker_profiles (user_id, display_name, speaker_embedding, is_active, updated_at)
        VALUES (%s, %s, %s::vector, TRUE, CURRENT_TIMESTAMP)
        ON CONFLICT (user_id) DO UPDATE
        SET display_name = EXCLUDED.display_name,
            speaker_embedding = EXCLUDED.speaker_embedding,
            is_active = TRUE,
            updated_at = CURRENT_TIMESTAMP;
    """
    try:
        with get_db_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id, display_name, embedding))
            conn.commit()
            logger.info(f"[SpeakerRepository] 화자 프로필 등록/갱신 완료 (user: {user_id}, name: {display_name})")
            return True
    except Exception as e:
        logger.error(f"[SpeakerRepository Error] 화자 프로필 등록/갱신 실패: {e}")
        return False


def get_active_speaker_embeddings() -> List[SpeakerEmbeddingRecord]:
    """
    1:N 코사인 유사도 비교 대상인 활성(is_active=TRUE) 화자 임베딩 전체를 조회한다.
    SpeakerService가 이 결과를 메모리에 캐싱해 매 발화마다 DB를 조회하지 않도록 한다.
    """
    query = """
        SELECT user_id, display_name, speaker_embedding
        FROM speaker_profiles
        WHERE is_active = TRUE;
    """
    try:
        with get_db_connection() as conn:
            register_vector(conn)
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return [
                    SpeakerEmbeddingRecord(
                        user_id=row[0], display_name=row[1], embedding=_to_embedding_list(row[2])
                    )
                    for row in rows
                ]
    except Exception as e:
        logger.error(f"[SpeakerRepository Error] 활성 화자 프로필 목록 조회 실패: {e}")
        return []


def deactivate_speaker_profile(user_id: str) -> None:
    """
    등록된 화자를 1:N 식별 대상에서 제외한다 (소프트 삭제, 물리 DELETE 없음).
    """
    query = """
        UPDATE speaker_profiles
        SET is_active = FALSE, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = %s;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id,))
            conn.commit()
            logger.info(f"[SpeakerRepository] 화자 프로필 비활성화 완료 (user: {user_id})")
    except Exception as e:
        logger.error(f"[SpeakerRepository Error] 화자 프로필 비활성화 실패: {e}")


# --- [Admin Dashboard] web/app.py 전용 조회/관리 메서드 ---
# 대시보드 쓰기 메서드는 UI에 성공/실패를 즉시 안내해야 하므로 영향받은 행 수 기준 bool을 반환한다.


def list_speaker_profiles(include_inactive: bool = True) -> List[SpeakerProfile]:
    """
    관리 화면용 화자 프로필 목록을 조회한다 (임베딩 컬럼은 전송량 절약을 위해 제외).
    활성 화자를 먼저, 그 안에서는 등록순으로 정렬한다.
    """
    query = """
        SELECT user_id, display_name, is_active, created_at, updated_at
        FROM speaker_profiles
        {where}
        ORDER BY is_active DESC, created_at ASC;
    """.format(where="" if include_inactive else "WHERE is_active = TRUE")
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query)
                rows = cursor.fetchall()
                return [
                    SpeakerProfile(
                        user_id=row[0], display_name=row[1], is_active=row[2], created_at=row[3], updated_at=row[4]
                    )
                    for row in rows
                ]
    except Exception as e:
        logger.error(f"[SpeakerRepository Error] 화자 프로필 목록 조회 실패: {e}")
        return []


def update_display_name(user_id: str, display_name: str) -> bool:
    """화자의 표시 이름(호칭)만 변경한다. 대상 행이 없으면 False."""
    query = """
        UPDATE speaker_profiles
        SET display_name = %s, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = %s;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (display_name, user_id))
                updated = cursor.rowcount > 0
            conn.commit()
            logger.info(f"[SpeakerRepository] 표시 이름 변경 (user: {user_id}, name: {display_name}, ok: {updated})")
            return updated
    except Exception as e:
        logger.error(f"[SpeakerRepository Error] 표시 이름 변경 실패: {e}")
        return False


def set_speaker_active(user_id: str, is_active: bool) -> bool:
    """화자를 1:N 식별 대상에 포함(재활성화)하거나 제외(비활성화)한다. 대상 행이 없으면 False."""
    query = """
        UPDATE speaker_profiles
        SET is_active = %s, updated_at = CURRENT_TIMESTAMP
        WHERE user_id = %s;
    """
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (is_active, user_id))
                updated = cursor.rowcount > 0
            conn.commit()
            logger.info(f"[SpeakerRepository] 활성 상태 변경 (user: {user_id}, is_active: {is_active}, ok: {updated})")
            return updated
    except Exception as e:
        logger.error(f"[SpeakerRepository Error] 활성 상태 변경 실패: {e}")
        return False


def delete_speaker_profile(user_id: str) -> bool:
    """
    화자 프로필(목소리 임베딩) 행을 영구 삭제한다. 대시보드에서 명시적 확인을 거친 경우에만 호출된다.
    ⚠️ 해당 user_id의 장기 기억(user_long_term_memory)과 프로필 요약은 건드리지 않는다
       (동일 user_id로 재등록 시 기존 기억이 그대로 이어지도록 보존).
    """
    query = "DELETE FROM speaker_profiles WHERE user_id = %s;"
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute(query, (user_id,))
                deleted = cursor.rowcount > 0
            conn.commit()
            logger.info(f"[SpeakerRepository] 화자 프로필 삭제 (user: {user_id}, ok: {deleted})")
            return deleted
    except Exception as e:
        logger.error(f"[SpeakerRepository Error] 화자 프로필 삭제 실패: {e}")
        return False
