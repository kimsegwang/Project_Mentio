"""
server/services/intent_service.py
발화 텍스트를 분석하여 초고속 텍스트 전용 대화인지, 카메라 프레임 동봉 대화인지,
혹은 LLM 호출 없이 즉답 가능한 룰 기반 질의(시간/날짜: 로컬 시계, 날씨: 캐시된 날씨 API)인지
판별하는 서비스.
"""
import logging
import re
from datetime import datetime
from typing import Tuple

from server.schemas.action import EmotionType, LLMResponse, TriggerType
from server.services.time_service import format_date_phrase, format_time_phrase

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

  # 2. [룰 기반] 날짜/요일 질의 패턴 (VOICE_TIME_RULE로 함께 라우팅)
  # "며칠 걸려", "며칠 전에"(기간/과거 시점) 오탐 방지를 위해 뒤따르는 기간 표현은 제외
  DATE_QUERY_PATTERNS = [
      r"(오늘|지금|현재)\s*(이\s*)?(며칠|몇\s*일)(?!\s*(걸|동안|전|후|뒤|째|만))",
      r"몇\s*월\s*(며칠|몇\s*일)",
      r"(며칠|몇\s*일)\s*(이야|이지|이니|인가|이에요|예요|입니까)",
      r"(무슨|몇)\s*요일(?!\s*에)",
      r"날짜\s*(좀\s*)?(알려|말해|뭐)",
      r"(오늘|지금|현재)\s*날짜",
  ]
  # "내일 무슨 요일이야"처럼 오늘이 아닌 날을 묻는 날짜 질의는 오늘 날짜로 즉답하면 틀리므로
  # 룰에서 제외하고, 현재 일시가 주입된 LLM(VOICE_CHAT)이 계산하도록 넘긴다.
  RELATIVE_DAY_PATTERN = r"(내일|어제|모레|그제|그저께|다음\s*주|지난\s*주|저번\s*주)"

  # 2. [룰 기반] 날씨 질의 패턴 (감지 시 LLM 호출 없이 캐시된 날씨 API 결과로 즉답)
  # "비 와서 우울해", "비 오는 날 좋아"처럼 비를 서술하는 일상 발화는 제외
  WEATHER_QUERY_PATTERNS = [
      r"날씨",
      r"비\s*(와|와요|오니|오나|올까|오려나|오고\s*있|내려|내리니|내리나)(?![서도는])",
      r"눈\s*(와|와요|오니|오나|올까|오려나|오고\s*있|내려|내리니|내리나)(?![서도는])",
      r"우산\s*(챙겨|필요|가져)",
      r"(기온|온도)\s*(몇|어때|알려|얼마)",
      r"(밖|바깥)\s*(에\s*)?(추워|더워|춥|덥)",
      r"(지금|오늘|밖|바깥)\s*몇\s*도",
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
    self.date_patterns = [
        re.compile(p) for p in self.DATE_QUERY_PATTERNS
    ]
    self.relative_day_regex = re.compile(self.RELATIVE_DAY_PATTERN)
    self.weather_patterns = [
        re.compile(p) for p in self.WEATHER_QUERY_PATTERNS
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

    # [2단계] 룰 기반 시간/날짜 질의 검사 -> LLM 호출 없이 로컬 시계로 즉시 응답
    if self._is_time_query(target_text) or self._is_date_query(target_text):
      logger.info(
          f"[Intent] Rule-based Time/Date Query -> Instant Local Clock: '{text}'"
          f" (Cleaned: '{target_text}')"
      )
      return TriggerType.VOICE_TIME_RULE.value, False

    # [2단계] 룰 기반 날씨 질의 검사 -> LLM 호출 없이 캐시된 날씨 정보로 즉시 응답
    if any(p.search(target_text) for p in self.weather_patterns):
      logger.info(
          f"[Intent] Rule-based Weather Query -> Cached Weather: '{text}'"
          f" (Cleaned: '{target_text}')"
      )
      return TriggerType.VOICE_WEATHER_RULE.value, False

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

  def _is_time_query(self, text: str) -> bool:
    return any(p.search(text) for p in self.time_patterns)

  def _is_date_query(self, text: str) -> bool:
    if self.relative_day_regex.search(text):
      return False
    return any(p.search(text) for p in self.date_patterns)

  def build_time_response(self, text: str = "") -> LLMResponse:
    """
    로컬 시스템 시계(datetime.now()) 기반으로 LLM 호출 없이 즉시 시간/날짜 안내 응답을 생성한다.
    발화에 날짜 질의만 있으면 날짜, 시간+날짜가 함께 있으면 복합 응답, 그 외(시간 질의 또는
    text 미전달)는 시간 안내로 응답한다.
    """
    now = datetime.now()
    clean_text = re.sub(r"[^\w\s]", " ", text or "")
    wants_date = self._is_date_query(clean_text)
    wants_time = self._is_time_query(clean_text) or not wants_date

    if wants_date and wants_time:
      speech = f"오늘은 {format_date_phrase(now)}, 지금은 {format_time_phrase(now)}이야!"
    elif wants_date:
      speech = f"오늘은 {format_date_phrase(now)}이야!"
    else:
      speech = f"지금은 {format_time_phrase(now)}이야!"
    return LLMResponse(emotion=EmotionType.HAPPY, speech=speech)


intent_service = IntentService()