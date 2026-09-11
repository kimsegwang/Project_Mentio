"""
server/services/intent_service.py
발화 텍스트를 분석하여 초고속 텍스트 전용 대화인지, 카메라 프레임 동봉 대화인지 판별하는 서비스.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class IntentService:
    # 1. 호출어 패턴 (쉼표, 느낌표, 공백까지 모두 제거)
    WAKE_WORDS_PATTERN = r"^(멘티오야|멘티아야|멘티오|멘티아|안녕\s*멘티오)[,\s!.~]*"

    # 2. 사진을 볼 필요 없는 명백한 일상어/기능 패턴 (문장 어디서든 매칭)
    EXPLICIT_CHAT_PATTERNS = [
        r"(안녕|반가워|잘자|좋은\s*아침|수고했어|바빠|뭐해)",
        r"(왔어|나왔어|다녀왔어|왔다|나\s*왔)",
        r"(날씨|시간|몇\s*시|날짜|요일)\s*(알려줘|어때|뭐야|궁금해)",
        r"(노래|음악)\s*(틀어|불러)",
        r"(고마워|감사합니다|잘했어|사랑해|이름이\s*뭐야)",
    ]

    def __init__(self):
        self.wake_regex = re.compile(self.WAKE_WORDS_PATTERN)
        self.explicit_chat = [re.compile(p) for p in self.EXPLICIT_CHAT_PATTERNS]
        logger.info("IntentService initialized.")

    def analyze_voice_intent(self, text: str) -> Tuple[str, bool]:
        if not text or not text.strip():
            return "VOICE_CHAT", False

        # 특수기호 및 여백 1차 정리 (마침표, 쉼표 등 제거)
        clean_text = re.sub(r"[^\w\s]", " ", text.strip())

        # 호출어("멘티오", "멘티오야" 등) 걷어내기
        trimmed_text = self.wake_regex.sub("", clean_text).strip()
        target_text = trimmed_text if trimmed_text else clean_text

        # 1. 일상 대화 패턴 검사 -> VOICE_CHAT
        if any(p.search(target_text) for p in self.explicit_chat):
            logger.info(f"[Intent] Pure Text Chat: '{text}' (Cleaned: '{target_text}')")
            return "VOICE_CHAT", False

        # 2. 그 외 시각 요구/애매한 문맥은 사진 동봉 -> VOICE_VISION
        logger.info(f"[Intent] Vision Attached: '{text}'")
        return "VOICE_VISION", True


intent_service = IntentService()