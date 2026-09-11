import os
import sys
import time
import threading
import cv2

# 프로젝트 루트 경로 등록
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from server.repositories.connection import init_db_pool, close_db_pool
from server.schemas.action import RobotAction
from server.services.brain_service import BrainService
from server.services.vision_service import VisionService
from server.services.audio_listener_service import audio_listener_service
from server.workers.ai_worker import AIWorker

# 전역 공유 상태
latest_frame = None
frame_lock = threading.Lock()
stop_event = threading.Event()
flash_trigger_time = 0.0


def audio_listener_worker(ai_worker: AIWorker, vision_service: VisionService):
    global flash_trigger_time
    print("[AudioWorker] 음성 감지 리스너 스레드 시작.")
    
    while not stop_event.is_set():
        audio_data = audio_listener_service.listen_phrase()
        
        if stop_event.is_set():
            break

        if audio_data is not None:
            # 📸 발화 종료 감지 즉시 화면 플래시 트리거 발동
            flash_trigger_time = time.time()
            print("\n📸 [찰칵!] 발화 종료 감지 -> 현재 웹캠 프레임 캡처 완료!")

            pil_snapshot = None
            with frame_lock:
                if latest_frame is not None:
                    pil_snapshot = vision_service.prepare_snapshot_for_vlm(latest_frame)

            threading.Thread(
                target=ai_worker.process_voice_interaction,
                args=(audio_data, pil_snapshot),
                daemon=True
            ).start()

        time.sleep(0.05)


def run_mentio_engine():
    global latest_frame

    print("=" * 65)
    print(" Project Sentio (Mentio) - Integrated Multimodal Engine")
    print("=" * 65)

    # 1. DB 커넥션 풀 초기화
    init_db_pool()

    # 2. 서비스 및 워커 레이어 인스턴스화
    vision_service = VisionService()
    brain_service = BrainService()
    ai_worker = AIWorker(brain_service_instance=brain_service)
    ai_worker.start()

    # 3. 비전 카메라 스트림 오픈
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[Engine Error] 카메라(Index: {settings.CAMERA_INDEX})를 열 수 없습니다.")
        ai_worker.stop()
        close_db_pool()
        return

    # 4. 음성 리스너 스레드 가동
    stop_event.clear()
    audio_thread = threading.Thread(
        target=audio_listener_worker,
        args=(ai_worker, vision_service),
        daemon=True
    )
    audio_thread.start()

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

    print("\n[안내] 제어 방식:")
    print(" - 음성 대화: 마이크에 언제든 발화 (예: '안녕', '이거 봐봐')")
    print(" - 양손 하트: 제스처 감정 반응")
    print(" - 's' 키: 수동 480p 스냅샷 VLM 분석")
    print(" - 'q' 키: 시스템 안전 종료\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # A. 최신 프레임 스냅샷 버퍼 갱신 (Thread-safe)
            with frame_lock:
                latest_frame = frame.copy()

            current_time = time.time()

            # B. 실시간 비전 처리 (30fps 무중단 제스처 감지)
            frame, is_heart = vision_service.process_gesture(frame)

            # C. 제스처 트리거 검사
            if is_heart and not is_processing and (current_time - last_event_time > settings.COOLDOWN_SECONDS):
                last_event_time = current_time
                is_processing = True
                status_msg = "Analyzing Heart Gesture..."
                prompt = "사용자가 양손으로 하트 제스처를 보냈습니다. 기쁨과 감사의 애정 표현을 담아 다정하게 반응하세요."
                ai_worker.submit_task("GESTURE", prompt, [prompt])

            # D. 키보드 인터럽트 처리
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and not is_processing:
                if current_time - last_event_time > settings.COOLDOWN_SECONDS:
                    last_event_time = current_time
                    is_processing = True
                    status_msg = "Analyzing Snapshot VLM..."

                    pil_img = vision_service.prepare_snapshot_for_vlm(frame)
                    prompt = "로봇 정면 카메라에 포착된 사용자와 주변 상황을 보고 1~2문장의 다정한 친구 말투로 요약해줘."
                    ai_worker.submit_task("SNAPSHOT", prompt, [pil_img, prompt])

            # E. 백그라운드 워커 결과 논블로킹 폴링
            result = ai_worker.poll_result()
            if result is not None:
                action, trigger_type, latency = result
                current_action = action
                is_processing = False
                status_msg = ""
                print(f"[Engine] 액션 반영 완료: [{current_action.emotion}] \"{current_action.speech}\"")

            # F. UI 오버레이 렌더링
            if is_processing:
                cv2.rectangle(frame, (10, 10), (450, 45), (0, 140, 255), -1)
                cv2.putText(frame, f"[AI Thinking] {status_msg}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            h_frame, w_frame, _ = frame.shape
            cv2.rectangle(frame, (0, h_frame - 60), (w_frame, h_frame), (30, 30, 30), -1)

            # 📸 캡처 순간 0.3초 동안 화면 테두리 플래시 및 텍스트 팝업
            if time.time() - flash_trigger_time < 0.35:
                # 화면 전체에 옅은 흰색 플래시 효과
                white_overlay = frame.copy()
                white_overlay[:] = (255, 255, 255)
                frame = cv2.addWeighted(frame, 0.6, white_overlay, 0.4, 0)
                # 찰칵 안내 텍스트
                cv2.putText(frame, "SNAPSHOT CAPTURED!", (w_frame // 2 - 160, h_frame // 2),
                            cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 255), 2)

            if is_processing:
                cv2.rectangle(frame, (10, 10), (450, 45), (0, 140, 255), -1)
                cv2.putText(frame, f"[AI Thinking] {status_msg}", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

            # LED 인디케이터 시각화
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
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        ai_worker.stop()
        close_db_pool()
        print("[Shutdown] 모든 리소스(카메라, 스레드, 워커, DB 커넥션)가 안전하게 해제되었습니다.")


if __name__ == "__main__":
    run_mentio_engine()