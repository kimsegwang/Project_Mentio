"""
server/services/intent_service.py
1차 로컬 패턴 매칭과 2차 LLM 시맨틱 라우팅을 결합한 이중화 의도 분석 서비스.
"""
import re
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class IntentService:
    # 1차: 명백하게 카메라 촬영/시각을 요구하는 확정 패턴 (중간 수식어 허용)
    EXPLICIT_VISION_PATTERNS = [
        r"사진\s*(좀\s*)?찍어",                                  # "사진 찍어줘", "사진 좀 찍어봐"
        r"(앞|주변|여기|저기)\s*(좀\s*|한\s*번\s*)?(봐|보여|봐봐)",    # "주변 봐봐", "앞에 좀 봐", "여기 한번 봐"
        r"(내\s*얼굴|표정|옷)\s*(좀\s*)?어때\s*보여",             # "내 표정 어때 보여?"
        r"(이거|저거|저런\s*거|이런\s*거).{0,10}(봐|보여|봐봐|봐볼래)", # "저거 한번 봐봐", "저런 거 한번 봐볼래"
        r"카메라\s*(켜|봐)",                                     # "카메라 켜봐"
    ]

    # 1차: 명백한 일상 텍스트 대화 (LLM 판별 건너뜀)
    EXPLICIT_CHAT_PATTERNS = [
        r"^(안녕|반가워|잘자|좋은\s*아침|수고했어)",
        r"(날씨|시간|몇\s*시|날짜)\s*(알려줘|어때|뭐야)",
        r"(노래|음악)\s*(틀어|불러)",
    ]

    def __init__(self):
        self.explicit_vision = [re.compile(p) for p in self.EXPLICIT_VISION_PATTERNS]
        self.explicit_chat = [re.compile(p) for p in self.EXPLICIT_CHAT_PATTERNS]
        logger.info("IntentService initialized with dual-tier routing.")

    def _fast_local_check(self, text: str) -> Tuple[bool, bool]:
        """
        로컬 패턴으로 즉시 확정 가능한지 검사합니다.
        """
        # 명백한 비전 요구
        if any(p.search(text) for p in self.explicit_vision):
            return True, True
        
        # 명백한 일반 대화
        if any(p.search(text) for p in self.explicit_chat):
            return True, False

        # 판단 유보
        return False, False

    def analyze_voice_intent(self, text: str, llm_classifier=None) -> Tuple[str, bool]:
        """
        1차 로컬 패턴 검사 후, 미확정 문장은 2차 LLM 시맨틱 라우팅을 수행합니다.
        """
        if not text or not text.strip():
            return "VOICE_CHAT", False

        normalized_text = text.strip()

        # 1차: 로컬 초고속 패턴 검사 (< 1ms)
        is_determined, needs_vision = self._fast_local_check(normalized_text)
        if is_determined:
            trigger_type = "VOICE_VISION" if needs_vision else "VOICE_CHAT"
            logger.info(f"[Tier 1 Fast Match] '{normalized_text}' -> {trigger_type}")
            return trigger_type, needs_vision

        # 2차: LLM 시맨틱 라우팅
        if llm_classifier:
            try:
                needs_vision = llm_classifier(normalized_text)
                trigger_type = "VOICE_VISION" if needs_vision else "VOICE_CHAT"
                logger.info(f"[Tier 2 LLM Match] '{normalized_text}' -> {trigger_type}")
                return trigger_type, needs_vision
            except Exception as e:
                logger.error(f"Tier 2 LLM routing failed: {e}. Falling back to default.")

        # 기본 fallback: 불필요한 카메라 작동 방지를 위해 CHAT
        return "VOICE_CHAT", False


intent_service = IntentService()