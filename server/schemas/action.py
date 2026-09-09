from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional

class TriggerType(str, Enum):
    PIR = "PIR"
    TOUCH = "TOUCH"
    PERIODIC = "PERIODIC"
    VOICE_CHAT = "VOICE_CHAT"
    VOICE_VISION = "VOICE_VISION"


class EmotionType(str, Enum):
    # 기본 대기
    NEUTRAL = "NEUTRAL"
    
    # 성격 및 대화 반응
    HAPPY = "HAPPY"
    PROUD = "PROUD"
    CURIOUS = "CURIOUS"
    SAD = "SAD"
    ANGRY = "ANGRY"
    
    # 물리 센서 반응
    HEART_EYES = "HEART_EYES"
    SURPRISED = "SURPRISED"
    SCARED = "SCARED"
    DIZZY = "DIZZY"
    
    # 환경 및 유틸리티
    TIRED = "TIRED"
    SLEEPING = "SLEEPING"
    FOCUS = "FOCUS"
    
    # 시스템 예외
    ERROR = "ERROR"


class LLMResponse(BaseModel):
    """
    Gemini VLM이 순수하게 추론하는 데이터 규격
    - RGB 필드를 제외하여 토큰 수 절약 및 레이턴시 단축
    """
    emotion: EmotionType = Field(
        description="상황에 알맞은 감정 (HAPPY, PROUD, CURIOUS, SAD, ANGRY, HEART_EYES, SURPRISED, TIRED 등)"
    )
    speech: str = Field(
        description="TTS로 출력할 1~2문장의 친근한 한국어 구어체 대사"
    )


class RobotAction(BaseModel):
    """
    ESP32-S3 전송 및 DB interaction_logs 적재용 완성형 DTO
    """
    emotion: EmotionType = Field(description="표정 및 눈동자 애니메이션 키")
    speech: str = Field(default="", description="TTS 출력 대사 (센서 단독 반응 시 빈 문자열 가능)")
    led_rgb: list[int] = Field(description="WS2812B RGB 삼원색 값 [R, G, B]")
    duration: float = Field(
        default=4.0,
        description="해당 액션 유지 시간(초). 0.0이면 외부 인터럽트 전까지 유지"
    )


FALLBACK_ACTION = RobotAction(
    emotion=EmotionType.ERROR,
    speech="생각이 조금 엉켰어요. 잠시 후에 다시 말해줘!",
    led_rgb=[255, 30, 0],
    duration=3.0
)

IDLE_ACTION = RobotAction(
    emotion=EmotionType.NEUTRAL,
    speech="",
    led_rgb=[100, 100, 100],
    duration=0.0
)