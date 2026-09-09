"""
server/services/intent_service.py
입력 텍스트를 분석하여 사용자의 의도 및 시각 정보(카메라 스냅샷) 필요 여부를 판별하는 서비스.
"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class IntentService:
    # 시각 동반 대화(VOICE_VISION) 트리거 키워드 정의
    VISION_TRIGGER_KEYWORDS = [
        "봐", "봐봐", "보여", "사진", "표정", 
        "얼굴", "주변", "이거", "저거", "어때 보여", "카메라"
    ]

    def __init__(self):
        logger.info("IntentService initialized.")

    def analyze_voice_intent(self, text: str) -> Tuple[str, bool]:
        """
        음성 인식 텍스트를 분석하여 음성 트리거 유형과 카메라 스냅샷 필요 여부를 결정합니다.

        Args:
            text: STT로 변환된 사용자 발화 문자열

        Returns:
            Tuple[str, bool]: (trigger_type, needs_vision)
                - ("VOICE_VISION", True): 시각적 맥락 필요 (스냅샷 1장 캡처)
                - ("VOICE_CHAT", False): 일반 음성 대화 (카메라 미가동)
        """
        if not text or not text.strip():
            return "VOICE_CHAT", False

        cleaned_text = text.replace(" ", "")
        needs_vision = any(keyword in cleaned_text for keyword in self.VISION_TRIGGER_KEYWORDS)

        trigger_type = "VOICE_VISION" if needs_vision else "VOICE_CHAT"
        logger.info(f"Intent Analyzed: text='{text}' -> trigger='{trigger_type}', needs_vision={needs_vision}")
        
        return trigger_type, needs_vision


# Spring Bean 스타일 전역 싱글톤 등록
intent_service = IntentService()