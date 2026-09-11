import os
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

# --- Gemini AI 설정 ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("[Config Error] 환경 변수에 GEMINI_API_KEY가 설정되어 있지 않습니다.")

GEMINI_MODEL_NAME = "gemini-3.6-flash"
API_TIMEOUT_MS = 10000  # 구글 SDK 기준 10초
MAX_RETRIES = 1
COOLDOWN_SECONDS = 6.0

# --- 비전 처리 설정 ---
CAMERA_INDEX = 0
TARGET_IMAGE_WIDTH = 480  # VLM 전송용 가로 리사이징 규격

# --- PostgreSQL DB 설정 (SQLD 특화) ---
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 5432))
DB_NAME = os.getenv("DB_NAME", "mentio_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "postgres")

# 커넥션 풀 설정
DB_POOL_MIN_CONN = 1
DB_POOL_MAX_CONN = 5

# TTS & Audio Settings
TTS_VOICE_NAME = "ko-KR-SunHiNeural"
TTS_RATE = "+0%"
TTS_PITCH = "+0Hz"
AUDIO_REVERB_GUARD_SEC = 0.4