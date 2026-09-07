"""
Project Mentio - Vision to Gemini Brain Pipeline Integration Test
- tests/test_gesture.py의 HeartDetector를 연동
- 손하트(BIG_HEART) 감지 시 자동으로 Gemini Flash 모델 호출
- RobotAction JSON(표정, 대사, LED RGB) 응답 생성 및 화면 표시
"""

import os
import sys
import time
import json
import cv2
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# 프로젝트 루트 경로 설정 및 모듈 로드
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

from config.settings import settings

# test_gesture.py의 HeartDetector 임포트
try:
    from test_gesture import HeartDetector
except ImportError:
    from tests.test_gesture import HeartDetector


class RobotAction(BaseModel):
    emotion: str = Field(
        description="로봇 표정 상태 (HAPPY, SAD, ANGRY, HEART_EYES, NEUTRAL, THINKING)"
    )
    speech: str = Field(
        description="사용자에게 음성으로 전달할 한국어 답변 대사 (친근하고 귀여운 로봇 구어체, 1~2문장)"
    )
    led_rgb: list[int] = Field(
        description="FireBeetle 2 GPIO 5번에 연결된 WS2812B LED 색상 [R, G, B] (각 0~255)",
        min_length=3,
        max_length=3,
    )


def trigger_gemini_action(event_description: str, max_retries: int = 2) -> dict | None:
    """비전 제스처 감지 시 Gemini 두뇌를 호출 (503 발생 시 자동 재시도)"""
    if not settings.GEMINI_API_KEY:
        print("[오류] GEMINI_API_KEY가 설정되지 않았습니다.")
        return None

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    system_instruction = (
        "당신은 감정을 표현하는 탁상형 반려로봇 'Mentio'입니다. "
        "카메라 비전 시스템이 감지한 사용자의 행동/제스처 이벤트가 주어집니다. "
        "상황에 맞는 감정(emotion), 자연스럽고 애정 어린 한국어 대사 1~2문장(speech), "
        "로봇 무드등 RGB LED 색상(led_rgb)을 지정된 JSON 규격으로만 즉시 출력하세요."
    )

    prompt = f"[비전 이벤트 감지] {event_description}. 이에 알맞은 로봇 반응을 출력해줘."
    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=RobotAction,
        temperature=0.7,
        max_output_tokens=1000,
    )

    print(f"\n🧠 [Gemini 두뇌 추론 시작] 이벤트: '{event_description}'")

    for attempt in range(max_retries + 1):
        try:
            start_time = time.perf_counter()
            response = client.models.generate_content(
                model="gemini-flash-latest",
                contents=prompt,
                config=config,
            )
            elapsed_time = time.perf_counter() - start_time
            print(f"⏱️ [Gemini 응답 완료] 소요 시간: {elapsed_time:.2f}초")
            print(f"📦 [출력 데이터]: {response.text}")
            return json.loads(response.text)

        except Exception as e:
            if "503" in str(e) and attempt < max_retries:
                wait_time = 1.5 * (attempt + 1)
                print(f"⚠️ 구글 서버 트래픽 일시 과부하(503). {wait_time:.1f}초 후 자동 재시도합니다... ({attempt + 1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"⚠️ [API 호출 실패]: {e}")
                return None

    return None


def run_pipeline():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return

    detector = HeartDetector()
    win_name = "Mentio - Vision to Brain Pipeline"
    cv2.namedWindow(win_name)

    print("\n=======================================================")
    print("🎥 Mentio 비전 파이프라인 통합 가동")
    print("- 카메라 앞에서 양손으로 '정밀 손하트'를 만들어보세요.")
    print("- 't' 키를 눌러 강제 수동 트리거 테스트도 가능합니다.")
    print("- 'q' 키를 누르면 종료됩니다.")
    print("=======================================================\n")

    last_trigger_time = 0.0
    cooldown_seconds = 6.0  # API 중복 연타 방지 쿨다운 (6초)
    last_robot_speech = ""

    while cap.isOpened():
        # 1. 창 닫기(X) 버튼 감지: 창이 닫혔으면 즉시 루프 탈출
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 창 닫기 버튼이 눌려 프로그램을 종료합니다.")
            break

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        gesture, text, color, processed_frame = detector.process_frame(frame)

        current_time = time.perf_counter()
        is_triggered = False

        # 1. 제스처 자동 감지 트리거
        if gesture == "HEART_EYES":
            is_triggered = True

        # 키보드 입력 체크
        key = cv2.waitKey(1) & 0xFF
        if key == ord("t"):
            is_triggered = True
            print("⌨️ 수동 테스트 키 입력 감지!")
        elif key == ord("q"):
            print("🛑 'q' 키 입력으로 프로그램을 종료합니다.")
            break

        # 2. 키 입력 대기 중 사용자가 'X'를 눌렀는지 재확인
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        # 쿨다운 검증 후 Gemini API 호출
        if is_triggered:
            if current_time - last_trigger_time > cooldown_seconds:
                last_trigger_time = current_time
                cv2.putText(
                    processed_frame,
                    "Calling Gemini Brain...",
                    (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(win_name, processed_frame)
                cv2.waitKey(1)

                action = trigger_gemini_action("사용자가 카메라를 향해 양손으로 예쁜 손하트를 보냈습니다!")
                if action:
                    last_robot_speech = action.get("speech", "")
            else:
                remaining = int(cooldown_seconds - (current_time - last_trigger_time))
                text = f"{text} (Cooldown {remaining}s)"

        # 상태 텍스트 렌더링
        cv2.putText(
            processed_frame,
            text,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        # Gemini가 응답한 최근 대사 화면 하단 표시
        if last_robot_speech:
            cv2.putText(
                processed_frame,
                f"Mentio: {last_robot_speech[:35]}...",
                (20, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        cv2.imshow(win_name, processed_frame)

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()