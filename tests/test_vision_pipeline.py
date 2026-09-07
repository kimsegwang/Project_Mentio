"""
Project Mentio - Vision to Gemini Brain Pipeline Integration Test
1. 손하트(BIG_HEART) 감지: 실시간 제스처 트리거 -> 감정 반응
2. 상황 인지 질의 (스냅샷 VLM): 's' 키 입력 시 카메라 1회 스냅샷 분석 -> 상황 묘사 대사
"""

import json
import os
import re
import sys
import time
import cv2
from google import genai
from google.genai import types
from PIL import Image
from pydantic import BaseModel, Field

# 프로젝트 루트 경로 설정 및 모듈 로드
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)
sys.path.append(CURRENT_DIR)

from config.settings import settings

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


# Client를 1회만 초기화하여 세션 재사용 (지연 시간 단축)
_GENAI_CLIENT = None

def get_genai_client() -> genai.Client | None:
    global _GENAI_CLIENT
    if _GENAI_CLIENT is None and settings.GEMINI_API_KEY:
        _GENAI_CLIENT = genai.Client(
            api_key=settings.GEMINI_API_KEY,
            http_options={"timeout": 10000},  # Google 최소 제한 10초 적용
        )
    return _GENAI_CLIENT


def parse_robot_action_json(raw_text: str) -> dict | None:
    """모델이 반환한 텍스트에서 순수 JSON 본문만 정규식으로 안전하게 추출"""
    raw_text = raw_text.strip()

    # 1. 텍스트 내에서 첫 번째 '{' 부터 마지막 '}' 까지 추출
    json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    # 2. 마크다운 코드블록 제거 후 재시도
    clean_text = re.sub(r"^```json\s*", "", raw_text, flags=re.MULTILINE)
    clean_text = re.sub(r"^```\s*", "", clean_text, flags=re.MULTILINE).strip()
    try:
        return json.loads(clean_text)
    except Exception as e:
        print(f"⚠️ [JSON 파싱 최종 실패]: {e} | 원본: {raw_text[:100]}")
        return None


def call_gemini_action(contents: list | str, max_retries: int = 1) -> dict | None:
    """타임아웃(10초) 및 지수 백오프 적용 추론 함수"""
    client = get_genai_client()
    if not client:
        print("[오류] GEMINI_API_KEY가 설정되지 않았습니다.")
        return None

    system_instruction = (
        "당신은 탁상형 반려로봇 'Mentio'입니다. "
        "카메라로 본 사물을 1초 만에 스캔하여 즉시 반응하세요. "
        "사족이나 생각 과정 없이, 감정(emotion), 한국어 1문장 대사(speech), "
        "RGB 색상(led_rgb)을 JSON 규격으로만 즉시 출력하세요."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=RobotAction,
        temperature=0.2,
        max_output_tokens=400,
    )

    for attempt in range(max_retries + 1):
        try:
            start_time = time.perf_counter()
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=contents,
                config=config,
            )
            elapsed_time = time.perf_counter() - start_time
            print(f"⏱️ [Gemini 응답 완료] 소요 시간: {elapsed_time:.2f}초")
            print(f"📦 [출력 데이터]: {response.text}")

            parsed_data = parse_robot_action_json(response.text)
            if parsed_data:
                return parsed_data

            return json.loads(response.text.strip())

        except Exception as e:
            err_msg = str(e)
            elapsed_time = time.perf_counter() - start_time

            if "Deadline Exceeded" in err_msg or "timed out" in err_msg.lower() or "timeout" in err_msg.lower():
                print(f"⏱️ [타임아웃 차단] 응답이 10초를 초과하여 강제 종료했습니다. (소요: {elapsed_time:.1f}초)")
                break

            if "503" in err_msg and attempt < max_retries:
                print(f"⚠️ 일시 트래픽 과부하(503). 1초 후 재시도... ({attempt + 1}/{max_retries})")
                time.sleep(1.0)
            elif "429" in err_msg:
                print("⏳ [호출 한도 도달] API 쿼터 한도에 도달했습니다. 잠시 후 시도해 주세요.")
                break
            else:
                print(f"⚠️ [API 호출 실패]: {e}")
                return None

    # 서버 실패/타임아웃 시 기본 안내 반응 반환
    return {
        "emotion": "THINKING",
        "speech": "주변을 열심히 보고 있는데 지금은 생각이 조금 복잡해요!",
        "led_rgb": [200, 200, 0],
    }


def trigger_gesture_event() -> dict | None:
    """하트 제스처 감지 이벤트 처리"""
    print("\n🧠 [제스처 이벤트 발생] 손하트 감지 -> Gemini 추론")
    prompt = "[비전 제스처 감지] 사용자가 카메라를 향해 양손으로 예쁜 손하트를 보냈습니다. 이에 알맞은 로봇 반응을 출력해줘."
    return call_gemini_action(contents=prompt)


def trigger_snapshot_vlm(frame) -> dict | None:
    """스냅샷 전송 시 JPEG 압축 적용 (가로 480px 리사이징)"""
    print("\n📸 [VLM 상황 인지] 카메라 프레임 스냅샷 캡처 중...")

    h, w = frame.shape[:2]
    target_w = 480
    target_h = int(h * (target_w / w))
    resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)

    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)

    prompt = "앞에 있는 물건이나 사람의 상태를 한눈에 보고, 귀엽고 친근하게 한 문장으로 즉시 말해줘."
    return call_gemini_action(contents=[pil_image, prompt])


def run_pipeline():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return

    detector = HeartDetector()
    win_name = "Mentio - Vision & VLM Pipeline"
    cv2.namedWindow(win_name)

    print("\n=======================================================")
    print("🎥 Mentio 비전 & VLM 파이프라인 통합 가동")
    print("- [손하트]: 양손 정밀 손하트 취하기 (자동 감지)")
    print("- [상황 인지]: 's' 키 누르기 (카메라 1회 스냅샷 분석)")
    print("- [수동 하트]: 't' 키 누르기")
    print("- [종료]: 'q' 키 누르기 또는 창 닫기(X)")
    print("=======================================================\n")

    last_trigger_time = 0.0
    cooldown_seconds = 6.0
    last_robot_speech = ""

    while cap.isOpened():
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            print("🛑 창 닫기 감지: 프로그램을 종료합니다.")
            break

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        gesture, text, color, processed_frame = detector.process_frame(frame.copy())

        current_time = time.perf_counter()
        action_type = None

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("🛑 'q' 키 입력: 프로그램을 종료합니다.")
            break
        elif key == ord("s"):
            action_type = "SNAPSHOT"
        elif key == ord("t"):
            action_type = "GESTURE"
        elif gesture == "HEART_EYES":
            action_type = "GESTURE"

        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        if action_type:
            if current_time - last_trigger_time > cooldown_seconds:
                last_trigger_time = current_time

                status_banner = "Analyzing Snapshot VLM..." if action_type == "SNAPSHOT" else "Calling Gemini Brain..."
                cv2.putText(
                    processed_frame,
                    status_banner,
                    (25, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )
                cv2.imshow(win_name, processed_frame)
                cv2.waitKey(1)

                action = None
                if action_type == "SNAPSHOT":
                    action = trigger_snapshot_vlm(frame)
                elif action_type == "GESTURE":
                    action = trigger_gesture_event()

                if action:
                    last_robot_speech = action.get("speech", "")
            else:
                remaining = int(cooldown_seconds - (current_time - last_trigger_time))
                text = f"{text} (Cooldown {remaining}s)"

        cv2.putText(
            processed_frame,
            text,
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2,
        )

        cv2.putText(
            processed_frame,
            "[s] Snapshot VLM  |  [t] Heart Test  |  [q] Quit",
            (20, processed_frame.shape[0] - 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )

        if last_robot_speech:
            cv2.putText(
                processed_frame,
                f"Mentio: {last_robot_speech[:38]}...",
                (20, processed_frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

        cv2.imshow(win_name, processed_frame)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_pipeline()