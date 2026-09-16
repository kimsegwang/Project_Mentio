"""
server/services/intent_service.py
발화 텍스트를 분석하여 초고속 텍스트 전용 대화인지, 카메라 프레임 동봉 대화인지,
혹은 LLM 호출 없이 로컬 시계로 즉답 가능한 룰 기반 질의인지 판별하는 서비스.
"""
import logging
import re
from datetime import datetime
from typing import Tuple

from server.schemas.action import EmotionType, LLMResponse, TriggerType

logger = logging.getLogger(__name__)


class IntentService:
  # 1. 호출어 패턴 (조사 '는, 가, 를, 도, 야, 아' 모두 걷어냄)
  WAKE_WORDS_PATTERN = r"^(멘티오|멘티아)(야|아|는|가|를|도|랑)?\s*"

  # 2. [룰 기반] 시간 질의 패턴 (감지 시 LLM 호출 없이 로컬 시계로 즉답)
  # "몇 시간"(소요 시간 질문)과의 오탐을 막기 위해 '시' 바로 뒤에 '간'이 오면 제외
  TIME_QUERY_PATTERNS = [
      r"(지금|현재)\s*몇\s*시(?!간)",
      r"몇\s*시(?!간)\s*(야|니|지|예요|이야|입니까|인가|쯤)?",
      r"(지금|현재)\s*시간(?!\s*(있|돼|괜찮|나|맞))",
      r"시간\s*(좀\s*)?(알려|말해)\s*(줘|주세요|줄래)?",
  ]

  # 3. [핵심] 시각 거부/부정 패턴 (이게 걸리면 뒤도 안 돌아보고 VOICE_CHAT으로 보냄)
  # "보지 마", "보지마", "찍지 마", "찍지마", "보지 마봐", "눈 감아", "보지마라" 등 완벽 커버
  VISION_NEGATIVE_PATTERNS = [
      r"(보지\s*(마|마봐|말아|마라|마요|않))",
      r"(찍지\s*(마|말아|마라|마요|않))",
      r"(눈\s*(감아|가려|돌려|감고))",
      r"(카메라\s*(꺼|가려|보지\s*마))",
  ]

  # 4. 명시적 시각 요구 패턴 (부정어를 통과한 순수 시각 요청만 검사)
  EXPLICIT_VISION_PATTERNS = [
      r"(봐봐|봐줘|보여줘|보이니|보이지|바바)",
      r"(이거|이건|여기|앞에|내\s*손|얼굴|표정|옷)\s*(뭐야|어때|어때보여|맞춰|보여)",
      r"(무슨\s*색|색깔|색상|몇\s*개|어떤\s*모양)",
      r"(사진|스냅샷|캡처)\s*(찍어|해줘)",
  ]

  def __init__(self):
    self.wake_regex = re.compile(self.WAKE_WORDS_PATTERN)
    self.time_patterns = [
        re.compile(p) for p in self.TIME_QUERY_PATTERNS
    ]
    self.negative_patterns = [
        re.compile(p) for p in self.VISION_NEGATIVE_PATTERNS
    ]
    self.vision_patterns = [
        re.compile(p) for p in self.EXPLICIT_VISION_PATTERNS
    ]
    logger.info("IntentService initialized with Negative Exclusion.")

  def analyze_voice_intent(self, text: str) -> Tuple[str, bool]:
    if not text or not text.strip():
      return TriggerType.VOICE_CHAT.value, False

    clean_text = re.sub(r"[^\w\s]", " ", text.strip())
    target_text = self.wake_regex.sub("", clean_text).strip()
    if not target_text:
      target_text = clean_text

    # [2단계] 룰 기반 시간 질의 검사 -> LLM 호출 없이 로컬 시계로 즉시 응답
    if any(p.search(target_text) for p in self.time_patterns):
      logger.info(
          f"[Intent] Rule-based Time Query -> Instant Local Clock: '{text}'"
          f" (Cleaned: '{target_text}')"
      )
      return TriggerType.VOICE_TIME_RULE.value, False

    # [3단계] 시각 부정어 검사 -> 걸리면 즉시 텍스트 챗으로 탈출!
    if any(p.search(target_text) for p in self.negative_patterns):
      logger.info(
          f"[Intent] Vision Negated -> Fast Chat: '{text}' (Cleaned:"
          f" '{target_text}')"
      )
      return TriggerType.VOICE_CHAT.value, False

    # [4단계] 순수 시각 요구 검사 -> 사진 동봉
    if any(p.search(target_text) for p in self.vision_patterns):
      logger.info(
          f"[Intent] Vision Attached: '{text}' (Cleaned: '{target_text}')"
      )
      return TriggerType.VOICE_VISION.value, True

    # [5단계] 기본 일상 대화 -> 초고속 텍스트 챗
    logger.info(
        f"[Intent] Default Pure Text Chat: '{text}' (Cleaned: '{target_text}')"
    )
    return TriggerType.VOICE_CHAT.value, False

  @staticmethod
  def build_time_response() -> LLMResponse:
    """로컬 시스템 시계(datetime.now()) 기반으로 LLM 호출 없이 즉시 시간 안내 응답을 생성한다."""
    now = datetime.now()
    period = "오전" if now.hour < 12 else "오후"
    hour_12 = now.hour % 12
    hour_12 = 12 if hour_12 == 0 else hour_12
    speech = f"지금은 {period} {hour_12}시 {now.minute}분이야!"
    return LLMResponse(emotion=EmotionType.HAPPY, speech=speech)


intent_service = IntentService()