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

# --- RAG 장기 기억(Long-term Memory) 설정 ---
DEFAULT_USER_ID = "primary_user"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # 384차원, FastEmbed(ONNX) 로컬 추론
EMBEDDING_DIM = 384
RAG_TOP_K = 2  # 프롬프트 주입 시 유사도 상위 문장 개수 상한
RAG_SIMILARITY_THRESHOLD = 0.80  # 이 값(코사인 유사도, 1에 가까울수록 유사) 미만인 기억은 무관한 것으로 간주해 프롬프트 주입에서 제외
# 실기 테스트 결과 0.65에서는 "안녕", "너 이름이 뭐야?" 같은 단순 인사/일반 질문에도
# all-MiniLM-L6-v2 임베딩 특유의 이방성으로 무관한 과거 기억이 새어 들어왔고,
# 0.75에서도 "안녕."(마침표 첫 턴)이 여전히 새고, 0.78에서도 "안녕."이 0.78xx로 턱걸이 통과하는 것이
# 로그로 확인되어 0.80으로 재상향함. "안녕?"은 0.78 시점에 이미 정상 차단됨을 로그로 확인.

RAG_DEDUP_SIMILARITY_THRESHOLD = 0.85  # 저장 전 근접 중복(Deduplication) 판별 임계값.
# 주입용 RAG_SIMILARITY_THRESHOLD(0.80)와는 목적이 달라 별도 분리함:
# 이 값은 "이미 저장된 사실과 사실상 동일한 발화인가"를 판별하는 값.
# 초기값 0.92 -> 0.88 하향 조정 히스토리:
#   - 0.92: "나 커피 좋아해." vs "커피 좋아해."(단순 주어 생략) 유사도 0.89로 중복 미감지 -> 0.88로 하향
#   - 0.88: "사과 좋아해"/"나 사과 좋아해"(0.90), "나는 고양이 좋아해"(0.90)는 정상 스킵되었으나
#     조사 생략("고양이 좋아해") 등 0.88 미만 케이스는 중복 저장되는 것이 확인되어, 조사/어미
#     변형을 흡수하도록 0.85로 재하향함. 단, STT 오인식("다 구양이 좋아해", 0.78)처럼 문자 자체가
#     달라지는 경우는 0.85로도 흡수되지 않는 별개 문제이며, 이번 조정의 해결 대상이 아니다.