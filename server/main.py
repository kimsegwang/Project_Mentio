import os
import sys
import time
import cv2

# 프로젝트 루트 경로 등록 (모듈 import 경로 일치)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from server.repositories.connection import init_db_pool, close_db_pool
from server.schemas.action import RobotAction
from server.services.brain_service import BrainService
from server.services.vision_service import VisionService
from server.workers.ai_worker import AIWorker


def run_mentio_engine():
    print("=" * 65)
    print(" Project Sentio (Mentio) - Layered Architecture Engine")
    print("=" * 65)

    # 1. DB 커넥션 풀 초기화
    init_db_pool()

    # 2. 서비스 및 워커 레이어 인스턴스화 (DI 조립)
    vision_service = VisionService()
    brain_service = BrainService()
    ai_worker = AIWorker(brain_service=brain_service)
    ai_worker.start()

    # 3. 비전 카메라 스트림 오픈
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Engine Error] 카메라(Index: {settings.CAMERA_INDEX})를 열 수 없습니다.")
        ai_worker.stop()
        close_db_pool()
        return

    window_name = "Mentio Robot Engine (30fps Vision & Async AI)"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    current_action = RobotAction(
        emotion="NEUTRAL",
        speech="시스템이 정상 가동 중입니다.",
        led_rgb=[0, 150, 255]
    )
    is_processing = False
    status_msg = ""
    last_event_time = 0.0

    print("\n[안내] 제어 키:")
    print(" - 양손 하트: 제스처 감정 반응")
    print(" - 's' 키: 480p 카메라 스냅샷 VLM 분석 (카메라 멈춤 없음)")
    print(" - 'q' 키: 시스템 안전 종료\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()

            # A. 실시간 비전 처리 (30fps 무중단)
            frame, is_heart = vision_service.process_gesture(frame)

            # B. 제스처 트리거 검사
            if is_heart and not is_processing and (current_time - last_event_time > settings.COOLDOWN_SECONDS):
                last_event_time = current_time
                is_processing = True
                status_msg = "Analyzing Heart Gesture..."
                prompt = "사용자가 양손으로 하트 제스처를 보냈습니다. 기쁨과 감사의 애정 표현을 담아 다정하게 반응하세요."
                ai_worker.submit_task("GESTURE", prompt, [prompt])

            # C. 키보드 인터럽트 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not is_processing:
                if current_time - last_event_time > settings.COOLDOWN_SECONDS:
                    last_event_time = current_time
                    is_processing = True
                    status_msg = "Analyzing Snapshot VLM..."

                    # 480p 이미지 변환
                    pil_img = vision_service.prepare_snapshot_for_vlm(frame)
                    prompt = "로봇 정면 카메라에 포착된 사용자와 주변 상황을 보고 1~2문장의 다정한 친구 말투로 요약해줘."
                    ai_worker.submit_task("SNAPSHOT", prompt, [pil_img, prompt])
                else:
                    remain = settings.COOLDOWN_SECONDS - (current_time - last_event_time)
                    print(f"[Cooldown] 대기 중: {remain:.1f}초 남음")

            # D. 백그라운드 워커 결과 논블로킹 폴링
            result = ai_worker.poll_result()
            if result is not None:
                action, trigger_type, latency = result
                current_action = action
                is_processing = False
                status_msg = ""
                print(f"[Engine] 액션 반영 완료: [{current_action.emotion}] \"{current_action.speech}\"")

            # E. UI 오버레이 렌더링
            if is_processing:
                cv2.rectangle(frame, (10, 10), (450, 45), (0, 140, 255), -1)
                cv2.putText(frame, f"[AI Thinking] {status_msg}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            h_frame, w_frame, _ = frame.shape
            cv2.rectangle(frame, (0, h_frame - 60), (w_frame, h_frame), (30, 30, 30), -1)

            # FireBeetle GPIO 5 NeoPixel 색상 인디케이터
            r, g, b = current_action.led_rgb
            cv2.rectangle(frame, (15, h_frame - 48), (45, h_frame - 15), (b, g, r), -1)
            cv2.rectangle(frame, (15, h_frame - 48), (45, h_frame - 15), (255, 255, 255), 1)

            status_line = f"EMOTION: {current_action.emotion} | LED: [{r},{g},{b}]"
            cv2.putText(frame, status_line, (60, h_frame - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 2)

            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break

            cv2.imshow(window_name, frame)

    finally:
        # 안전한 자원 반납
        cap.release()
        cv2.destroyAllWindows()
        ai_worker.stop()
        close_db_pool()
        print("[Shutdown] 모든 리소스(카메라, 워커, DB 커넥션)가 안전하게 해제되었습니다.")


if __name__ == "__main__":
    run_mentio_engine()