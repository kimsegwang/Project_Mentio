from pydantic import BaseModel, Field


class RobotAction(BaseModel):
    """
    로봇 Mentio의 물리 인터랙션 및 표정/대사 제어 규격
    FireBeetle 2(ESP32-S3) 통신 및 DB 적재 공용 DTO
    """
    emotion: str = Field(
        description="로봇 표정 상태 (예: HAPPY, HEART_EYES, THINKING, NEUTRAL, SURPRISED)"
    )
    speech: str = Field(
        description="TTS로 출력할 한국어 구어체 대사 (1~2문장)"
    )
    led_rgb: list[int] = Field(
        description="FireBeetle GPIO 5 제어용 WS2812B RGB 삼원색 값 [R, G, B] (각 0~255)"
    )


# API 장애/타임아웃/429 시 로봇 정지를 방지하는 기본 Fallback 액션
FALLBACK_ACTION = RobotAction(
    emotion="THINKING",
    speech="생각이 조금 복잡해요! 잠시 후에 다시 이야기해 주세요.",
    led_rgb=[255, 180, 0]
)