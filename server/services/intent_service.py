"""
server/services/intent_service.py
발화 텍스트를 분석하여 초고속 텍스트 전용 대화인지, 카메라 프레임 동봉 대화인지 판별하는 서비스.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class IntentService:
    # 사진을 절대 볼 필요 없는 명백한 일상어/기능 패턴 (0ms 로컬 처리)
    EXPLICIT_CHAT_PATTERNS = [
        r"^(안녕|반가워|잘자|좋은\s*아침|수고했어)",
        r"(날씨|시간|몇\s*시|날짜)\s*(알려줘|어때|뭐야)",
        r"(노래|음악)\s*(틀어|불러)",
        r"(고마워|감사합니다|잘했어)",
    ]

    def __init__(self):
        self.explicit_chat = [re.compile(p) for p in self.EXPLICIT_CHAT_PATTERNS]
        logger.info("IntentService initialized.")

    def analyze_voice_intent(self, text: str) -> Tuple[str, bool]:
        """
        텍스트를 분석하여 시각 프레임 동봉 여부를 결정합니다.

        Returns:
            Tuple[trigger_type, needs_vision]
        """
        if not text or not text.strip():
            return "VOICE_CHAT", False

        normalized_text = text.strip()

        # 1. 명백한 일상어는 텍스트만 전송 (토큰 및 대역폭 절약)
        if any(p.search(normalized_text) for p in self.explicit_chat):
            logger.info(f"[Intent] Pure Text Chat: '{normalized_text}'")
            return "VOICE_CHAT", False

        # 2. 시각 요구 명령 및 애매한 모든 문맥은 사진 1장을 동봉해 VLM 1회 호출로 위임
        logger.info(f"[Intent] Vision Attached: '{normalized_text}'")
        return "VOICE_VISION", True


intent_service = IntentService()