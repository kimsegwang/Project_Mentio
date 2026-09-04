"""
Project Mentio - Gemini Brain & Action Schema Module
- Pydantic 기반 구조화된 JSON 강제 출력 (Structured Output)
- FireBeetle 2 ESP32-S3 제어 규격 정의 (표정, 대사, WS2812B RGB)
"""

import os
import sys
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 프로젝트 루트 경로 참조
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.settings import settings


class RobotAction(BaseModel):
    emotion: str = Field(
        description="로봇 표정 상태 (HAPPY, SAD, ANGRY, HEART_EYES, NEUTRAL, THINKING)"
    )
    speech: str = Field(
        description="사용자에게 음성으로 전달할 한국어 답변 대사 (친근한 로봇 구어체, 1~2문장)"
    )
    led_rgb: list[int] = Field(
        description="FireBeetle 2 GPIO 5번에 연결된 WS2812B LED 색상 [R, G, B] (각 0~255)",
        min_length=3,
        max_length=3
    )


def test_gemini_brain(prompt: str):
    if not settings.GEMINI_API_KEY:
        print("[오류] .env 파일에 GEMINI_API_KEY가 설정되지 않았습니다.")
        return

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    system_instruction = (
        "당신은 감정을 표현하는 탁상형 반려로봇 'Mentio'입니다. "
        "사용자의 말과 입력 상황에 반응하여 적절한 감정(emotion), 자연스러운 한국어 대사(speech), "
        "로봇 눈/가슴에 점등할 RGB LED 색상(led_rgb)을 항상 지정된 JSON 규격으로만 출력하세요."
    )

    print(f"질의 전송 중: '{prompt}'...")

    response = client.models.generate_content(
        model="gemini-3.6-flash",  # 현재 공식 기본 플래시 모델
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=RobotAction,
        ),
    )

    print("\n=== Gemini 정형 응답 ===")
    print(response.text)


if __name__ == "__main__":
    test_gemini_brain("안녕 멘티오! 오늘 기분이 어때?")