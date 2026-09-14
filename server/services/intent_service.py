"""
server/services/intent_service.py
발화 텍스트를 분석하여 초고속 텍스트 전용 대화인지, 카메라 프레임 동봉 대화인지 판별하는 서비스.
"""
import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


class IntentService:
  # 1. 호출어 패턴 (조사 '는, 가, 를, 도, 야, 아' 모두 걷어냄)
  WAKE_WORDS_PATTERN = r"^(멘티오|멘티아)(야|아|는|가|를|도|랑)?\s*"

  # 2. [핵심] 시각 거부/부정 패턴 (이게 걸리면 뒤도 안 돌아보고 VOICE_CHAT으로 보냄)
  # "보지 마", "보지마", "찍지 마", "찍지마", "보지 마봐", "눈 감아", "보지마라" 등 완벽 커버
  VISION_NEGATIVE_PATTERNS = [
      r"(보지\s*(마|마봐|말아|마라|마요|않))",
      r"(찍지\s*(마|말아|마라|마요|않))",
      r"(눈\s*(감아|가려|돌려|감고))",
      r"(카메라\s*(꺼|가려|보지\s*마))",
  ]

  # 3. 명시적 시각 요구 패턴 (부정어를 통과한 순수 시각 요청만 검사)
  EXPLICIT_VISION_PATTERNS = [
      r"(봐봐|봐줘|보여줘|보이니|보이지|바바)",
      r"(이거|이건|여기|앞에|내\s*손|얼굴|표정|옷)\s*(뭐야|어때|어때보여|맞춰|보여)",
      r"(무슨\s*색|색깔|색상|몇\s*개|어떤\s*모양)",
      r"(사진|스냅샷|캡처)\s*(찍어|해줘)",
  ]

  def __init__(self):
    self.wake_regex = re.compile(self.WAKE_WORDS_PATTERN)
    self.negative_patterns = [
        re.compile(p) for p in self.VISION_NEGATIVE_PATTERNS
    ]
    self.vision_patterns = [
        re.compile(p) for p in self.EXPLICIT_VISION_PATTERNS
    ]
    logger.info("IntentService initialized with Negative Exclusion.")

  def analyze_voice_intent(self, text: str) -> Tuple[str, bool]:
    if not text or not text.strip():
      return "VOICE_CHAT", False

    clean_text = re.sub(r"[^\w\s]", " ", text.strip())
    target_text = self.wake_regex.sub("", clean_text).strip()
    if not target_text:
      target_text = clean_text

    # [1단계] 시각 부정어 검사 -> 걸리면 즉시 텍스트 챗으로 탈출!
    if any(p.search(target_text) for p in self.negative_patterns):
      logger.info(
          f"[Intent] Vision Negated -> Fast Chat: '{text}' (Cleaned:"
          f" '{target_text}')"
      )
      return "VOICE_CHAT", False

    # [2단계] 순수 시각 요구 검사 -> 사진 동봉
    if any(p.search(target_text) for p in self.vision_patterns):
      logger.info(
          f"[Intent] Vision Attached: '{text}' (Cleaned: '{target_text}')"
      )
      return "VOICE_VISION", True

    # [3단계] 기본 일상 대화 -> 초고속 텍스트 챗
    logger.info(
        f"[Intent] Default Pure Text Chat: '{text}' (Cleaned: '{target_text}')"
    )
    return "VOICE_CHAT", False


intent_service = IntentService()