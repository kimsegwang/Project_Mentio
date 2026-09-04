"""
Project Mentio - Gemini Brain Latency & Stability Optimized
- gemini-flash-latest 모델 적용
- max_output_tokens 확보로 JSON 잘림 방지
"""

import os
import sys
import time
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

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
        "사용자의 말과 상황에 반응하여 적절한 감정(emotion), 짧고 명확한 한국어 대사 1~2문장(speech), "
        "RGB LED 색상(led_rgb)을 지정된 JSON 규격으로만 즉시 출력하세요. 서두 인사나 마크다운 없이 순수 JSON만 반환하세요."
    )

    print(f"질의 전송 중: '{prompt}'...")

    start_time = time.perf_counter()

    response = client.models.generate_content(
        model="gemini-flash-latest",
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=RobotAction,
            temperature=0.7,
            max_output_tokens=1000,  # JSON이 중간에 잘리지 않도록 충분히 확보
        ),
    )

    elapsed_time = time.perf_counter() - start_time

    print("\n=== Gemini 정형 응답 ===")
    print(response.text)
    print(f"\n[성능 측정] 총 응답 소요 시간: {elapsed_time:.2f}초")


if __name__ == "__main__":
    test_gemini_brain("안녕 멘티오! 오늘 기분이 어때?")