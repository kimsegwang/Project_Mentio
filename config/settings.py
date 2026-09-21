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

RAG_QUESTION_SIMILARITY_THRESHOLD = 0.68  # 사용자 발화가 의문문("~과일 좋아한다고 했지?")으로 판별될 때만 적용하는 완화된 주입 임계값.
# 서술문(RAG_SIMILARITY_THRESHOLD=0.80)과 그에 대응하는 질문 사이에는 로컬 임베딩(all-MiniLM-L6-v2)
# 특유의 비대칭성으로 유사도가 0.80 밑으로 떨어지는 현상이 실기 테스트로 확인되어, 의문문에 한해서만
# 임계값을 낮춰 재현율을 확보한다. RAG_TOP_K=2 상한이 이미 있어 과도한 주입 노이즈는 방어된다.

RAG_DEDUP_SIMILARITY_THRESHOLD = 0.85  # 저장 전 근접 중복(Deduplication) 판별 임계값.
# 주입용 RAG_SIMILARITY_THRESHOLD(0.80)와는 목적이 달라 별도 분리함:
# 이 값은 "이미 저장된 사실과 사실상 동일한 발화인가"를 판별하는 값.
# 초기값 0.92 -> 0.88 하향 조정 히스토리:
#   - 0.92: "나 커피 좋아해." vs "커피 좋아해."(단순 주어 생략) 유사도 0.89로 중복 미감지 -> 0.88로 하향
#   - 0.88: "사과 좋아해"/"나 사과 좋아해"(0.90), "나는 고양이 좋아해"(0.90)는 정상 스킵되었으나
#     조사 생략("고양이 좋아해") 등 0.88 미만 케이스는 중복 저장되는 것이 확인되어, 조사/어미
#     변형을 흡수하도록 0.85로 재하향함. 단, STT 오인식("다 구양이 좋아해", 0.78)처럼 문자 자체가
#     달라지는 경우는 0.85로도 흡수되지 않는 별개 문제이며, 이번 조정의 해결 대상이 아니다.

# --- RAG 장기 기억 모순 해결(Invalidation) 설정 ---
RAG_CONFLICT_CANDIDATE_THRESHOLD = 0.55  # 이 값 이상 RAG_DEDUP_SIMILARITY_THRESHOLD 미만인 기존 기억만
# "모순 후보"로 간주해 경량 LLM Reflection(BrainService.classify_memory_relation)에 전달한다.
# 순수 임베딩 유사도만으로는 "사과 좋아해"/"사과 싫어해"처럼 주제는 같고 극성만 반대인 문장을
# 구분하지 못하므로(오히려 유사도가 높게 나옴), 완전 무관(< 0.55)한 기억까지 매번 LLM에 태우지
# 않도록 후보 구간만 좁혀 호출 비용을 최소화한다. 초기값이며, Dedup 임계값과 동일하게 실기 로그
# 기반 반복 튜닝이 필요할 수 있다.
RAG_CONFLICT_TOP_K = 3  # 모순 판정 LLM 호출 시 전달할 후보 기억 개수 상한.

# --- Memory Summarization(장기 기억 요약) 설정 ---
PROFILE_SUMMARY_SOURCE_LIMIT = 50  # 프로필 요약 생성 시 반영할 활성 기억(is_active=TRUE) 최대 개수.
# 대화 턴과 무관한 배치/백그라운드 파이프라인이므로 RAG_TOP_K(2)처럼 엄격히 제한할 필요는 없으나,
# 사용자당 기억이 무한정 쌓였을 때 단일 LLM 호출 프롬프트가 과도하게 길어지는 것을 방지하는 상한이다.

PROFILE_SUMMARY_TRIGGER_COUNT = 5  # MemoryWriteWorker가 신규 기억을 이 개수만큼 누적 적재하면
# summarize_user_profile()을 자동으로 트리거한다. 근접 중복(Dedup)으로 스킵된 발화는 실제 INSERT가
# 아니므로 이 카운트에 포함하지 않는다. 기존 프로필 요약이 아예 없는 Cold Start 상태에서는
# 이 개수에 도달하기 전이라도 최초 신규 기억 적재 시점에 즉시 트리거한다.

# --- 화자 식별(Speaker Verification) 설정 ---
SPEAKER_VERIFICATION_ENABLED = os.getenv("SPEAKER_VERIFICATION_ENABLED", "true").lower() == "true"
# False로 끄거나, 기준 화자가 아직 등록되지 않은 경우(등록 파일 없음) SpeakerService.verify()는
# 안전하게 검증을 스킵(통과)한다. 즉 "미등록 = 차단"이 아니라 "미등록 = 무검증 통과"가 기본 동작이다.

SPEAKER_REFERENCE_EMBEDDING_PATH = os.getenv(
    "SPEAKER_REFERENCE_EMBEDDING_PATH",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "speaker_profiles", "primary_user.npy"),
)  # scripts/enroll_speaker.py가 기준 화자(primary_user) 임베딩을 저장하는 경로.

SPEAKER_VERIFICATION_THRESHOLD = float(os.getenv("SPEAKER_VERIFICATION_THRESHOLD", "0.65"))
# Resemblyzer 임베딩 기준 코사인 유사도 임계값. 이 값 미만이면 제3자 발화/TV 소리 등으로
# 판단해 파이프라인 진입을 차단한다.
# 튜닝 히스토리:
#   - 초기값 0.75 -> 실기 테스트에서 본인 목소리인데도 발화의 60% 이상이 차단되는 현상이
#     확인됨. 원인 조사 결과 검증 로그가 콘솔에 전혀 보이지 않아(logger 핸들러 미구성) 실측
#     유사도 점수를 확인할 수 없었던 것이 원인 파악을 늦췄고, 콘솔 핸들러를 명시 구성해 점수를
#     직접 확인해보니 정상 발화도 0.65~0.74 구간에 다수 분포함이 확인되어 0.68로 1차 하향.
#   - 0.68 -> 상세 로깅 반영 후 재측정한 정상 발화 점수 분포(0.799, 0.678, 0.686, 0.829,
#     0.703)에서 0.678이 0.68에 아깝게 컷오프되는 사례가 확인되어, 더 안정적인 여유폭
#     확보를 위해 0.65로 최종 하향함.

SPEAKER_SHORT_UTTERANCE_MAX_SEC = 1.0
# 이 길이(초) 이하의 발화는 "짧은 발화"로 간주해 SPEAKER_SHORT_UTTERANCE_THRESHOLD를 적용한다.
# Resemblyzer는 발화 길이가 짧을수록(예: "네", "응") 화자 특징을 충분히 추출하지 못해 동일
# 화자의 발화조차 유사도 점수가 급락하는 경향이 실기 로그로 확인되었다.

SPEAKER_SHORT_UTTERANCE_THRESHOLD = 0.60
# 짧은 발화(SPEAKER_SHORT_UTTERANCE_MAX_SEC 이하)에 한해서만 적용하는 완화된 임계값.
# 일반 발화 임계값(SPEAKER_VERIFICATION_THRESHOLD)과 분리해, 충분히 긴 발화에 대해서는
# 여전히 엄격한 기준을 유지한다.

SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC = 10.0
# 직전에 화자 검증을 통과(하드/소프트패스 무관)한 시점으로부터 이 시간(초) 이내에 들어온
# 발화는, 이번 발화의 유사도가 임계값에 미달하더라도 "같은 대화 세션이 이어지는 중"으로
# 간주해 소프트패스로 통과시킨다. 연속 대화 중 짧은 맞장구("어", "음")로 인해 정상 사용자의
# 세션이 중간에 끊기는 것을 방지하기 위함이다.