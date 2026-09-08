import logging
from server.repositories.connection import get_db_connection

logger = logging.getLogger(__name__)

# 인메모리 감정 프리셋 캐시
EMOTION_CACHE: dict[str, dict] = {}


def load_emotion_presets() -> dict[str, dict]:
    """
    서버 초기화 시 DB에서 emotion_presets 테이블을 조회하여 인메모리 캐시에 적재합니다.
    """
    global EMOTION_CACHE
    try:
        with get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT emotion, r, g, b, duration FROM emotion_presets;")
                rows = cur.fetchall()
                EMOTION_CACHE = {
                    row[0]: {
                        "rgb": [row[1], row[2], row[3]],
                        "duration": float(row[4])
                    }
                    for row in rows
                }
                print(f"[DB] 감정 프리셋 {len(EMOTION_CACHE)}종 캐시 적재 완료")
                return EMOTION_CACHE
    except Exception as e:
        print(f"[DB Error] 감정 프리셋 로드 실패: {e}")
        return EMOTION_CACHE


def get_preset_for_emotion(emotion: str) -> dict:
    """
    캐시에서 해당 감정의 RGB 및 기본 지속시간을 조회합니다.
    매핑 실패 시 기본 안전값(화이트 계열, 4.0초)을 반환합니다.
    """
    return EMOTION_CACHE.get(
        emotion, 
        {"rgb": [100, 100, 100], "duration": 4.0}
    )