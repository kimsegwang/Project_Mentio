"""
Project Mentio - Vision to Gemini Brain Pipeline Integration Test (Non-blocking Threaded)
1. 손하트(BIG_HEART) 감지: 실시간 제스처 트리거 -> 백그라운드 큐 전달
2. 상황 인지 질의 (스냅샷 VLM): 's' 키 입력 시 카메라 1회 스냅샷 -> 백그라운드 큐 전달
3. 메인 루프: 30fps 비전 프리뷰 유지 (화면 멈춤 현상 완전 제거)
"""

import json
import os
import queue
import re
import sys
import threading
import time
import cv2
from google import genai
from google.genai import types
from PIL import Image
from typing import List
from typing_extensions import TypedDict

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


class RobotAction(TypedDict):
    emotion: str  # 로봇 표정: HAPPY, SAD, ANGRY, HEART_EYES, NEUTRAL, THINKING
    speech: str   # 1~2문장의 한국어 구어체 대사
    led_rgb: List[int]  # [R, G, B] 각 0~255


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


def call_gemini_action(contents: list | str) -> dict:
    fallback_response = {
        "emotion": "THINKING",
        "speech": "주변을 열심히 보고 있는데 지금은 생각이 조금 복잡해요!",
        "led_rgb": [200, 200, 0],
    }

    client = get_genai_client()
    if not client:
        print("[오류] GEMINI_API_KEY가 설정되지 않았습니다.")
        return fallback_response

    system_instruction = (
        "당신은 탁상형 반려로봇 'Mentio'입니다. "
        "사용자의 제스처나 카메라 영상을 보고 반응하세요. "
        "사족 없이 지정된 JSON 형식으로만 즉시 출력하세요. "
        "emotion은 HAPPY, SAD, ANGRY, HEART_EYES, NEUTRAL 중 하나를 선택하고, "
        "speech는 친근한 한국어 1~2문장, led_rgb는 [R, G, B] 리스트여야 합니다."
    )

    config = types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=RobotAction,
        temperature=0.2,
        max_output_tokens=1500,
    )

    start_time = time.perf_counter()
    try:
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
        if "400" in err_msg:
            print(f"⚠️ [파라미터 오류 400]: {err_msg}")
        elif "429" in err_msg:
            print("⏳ [호출 한도 429] 쿼터 초과. 잠시 대기 필요.")
        else:
            print(f"⚠️ [API 예외]: {e}")
        return fallback_response


# =======================================================
# 백그라운드 VLM 추론 워커 루프
# =======================================================
def vlm_worker_thread(task_queue: queue.Queue, result_queue: queue.Queue):
    """
    메인 스레드로부터 작업(action_type, payload)을 전달받아
    동기식 Gemini 호출을 백그라운드에서 처리한 뒤 결과를 result_queue로 전달
    """
    while True:
        task = task_queue.get()
        if task is None:  # 종료 신호 수신
            task_queue.task_done()
            break

        action_type, payload = task
        action_result = None

        try:
            if action_type == "SNAPSHOT":
                # frame 리사이징 및 PIL Image 변환
                frame = payload
                h, w = frame.shape[:2]
                target_w = 480
                target_h = int(h * (target_w / w))
                resized_frame = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)

                prompt = "앞에 있는 물건이나 사람의 상태를 한눈에 보고, 귀엽고 친근하게 한 문장으로 즉시 말해줘."
                action_result = call_gemini_action(contents=[pil_image, prompt])

            elif action_type == "GESTURE":
                prompt = "[비전 제스처 감지] 사용자가 카메라를 향해 양손으로 예쁜 손하트를 보냈습니다. 이에 알맞은 로봇 반응을 출력해줘."
                action_result = call_gemini_action(contents=prompt)

        except Exception as e:
            print(f"❌ [워커 예외 발생]: {e}")
            action_result = {
                "emotion": "THINKING",
                "speech": "생각하는 중에 잠시 멍해졌어요!",
                "led_rgb": [200, 200, 0],
            }
        finally:
            if action_result:
                result_queue.put(action_result)
            task_queue.task_done()


def run_pipeline():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[오류] 웹캠을 열 수 없습니다.")
        return

    detector = HeartDetector()
    win_name = "Mentio - Vision & VLM Pipeline"
    cv2.namedWindow(win_name)

    # 비동기 통신용 스레드 및 큐 구성
    task_queue = queue.Queue(maxsize=1)   # 백프레셔 제어: 작업은 한 번에 1개만 대기
    result_queue = queue.Queue()          # 메인 스레드 결과 수신용
    
    worker = threading.Thread(
        target=vlm_worker_thread,
        args=(task_queue, result_queue),
        daemon=True
    )
    worker.start()

    print("\n=======================================================")
    print("🎥 Mentio 비전 & VLM 파이프라인 가동 (Non-blocking UI)")
    print("- [손하트]: 양손 정밀 손하트 취하기 (자동 감지)")
    print("- [상황 인지]: 's' 키 누르기 (카메라 1회 스냅샷 분석)")
    print("- [수동 하트]: 't' 키 누르기")
    print("- [종료]: 'q' 키 누르기 또는 창 닫기(X)")
    print("=======================================================\n")

    last_trigger_time = 0.0
    cooldown_seconds = 6.0
    last_robot_speech = ""
    current_emotion = "NEUTRAL"
    current_led = [100, 100, 100]
    is_processing = False  # VLM API 호출 진행 상태 플래그

    try:
        while cap.isOpened():
            # 1. 루프 시작 시점 창 닫힘 체크
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                print("🛑 창 닫기(X) 감지: 프로그램을 종료합니다.")
                break

            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            gesture, text, color, processed_frame = detector.process_frame(frame.copy())

            # 백그라운드 큐 수신
            try:
                res = result_queue.get_nowait()
                last_robot_speech = res.get("speech", "")
                current_emotion = res.get("emotion", "NEUTRAL")
                current_led = res.get("led_rgb", [100, 100, 100])
                is_processing = False
                result_queue.task_done()
            except queue.Empty:
                pass

            current_time = time.perf_counter()
            action_type = None

            # 2. 키 입력 및 X 버튼 재확인
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
                print("🛑 종료 감지: 파이프라인을 종료합니다.")
                break
            elif key == ord("s"):
                action_type = "SNAPSHOT"
            elif key == ord("t"):
                action_type = "GESTURE"
            elif gesture == "HEART_EYES":
                action_type = "GESTURE"

            # 3. 요청 전달 처리
            if action_type:
                if is_processing:
                    text = f"{text} (Brain Thinking...)"
                elif current_time - last_trigger_time > cooldown_seconds:
                    last_trigger_time = current_time
                    is_processing = True
                    payload = frame.copy() if action_type == "SNAPSHOT" else None
                    try:
                        task_queue.put_nowait((action_type, payload))
                    except queue.Full:
                        is_processing = False
                else:
                    remaining = int(cooldown_seconds - (current_time - last_trigger_time))
                    text = f"{text} (Cooldown {remaining}s)"

            # 4. UI 오버레이 렌더링
            if is_processing:
                cv2.putText(processed_frame, "Mentio Brain: THINKING...", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)

            cv2.putText(processed_frame, text, (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(processed_frame, "[s] Snapshot VLM  |  [t] Heart Test  |  [q] Quit", (20, processed_frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            if last_robot_speech:
                cv2.putText(processed_frame, f"[{current_emotion}] Mentio: {last_robot_speech[:35]}...", (20, processed_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

            bgr_led = (current_led[2], current_led[1], current_led[0])
            cv2.circle(processed_frame, (processed_frame.shape[1] - 40, 40), 16, bgr_led, -1)
            cv2.circle(processed_frame, (processed_frame.shape[1] - 40, 40), 18, (255, 255, 255), 2)

            # 5. 프레임 표시 직전 최종 가시성 체크 (다시 켜짐 원천 차단)
            if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) >= 1:
                cv2.imshow(win_name, processed_frame)

    finally:
        # 종료 플래그 전달 및 클린업
        try:
            task_queue.put_nowait(None)
        except queue.Full:
            pass
        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)  # 윈도우 OS의 이벤트 루프 잔여 버퍼 비우기


if __name__ == "__main__":
    run_pipeline()