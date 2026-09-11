"""
tests/test_voice_vision_e2e.py
자연스러운 인터랙션 순서: 음성 청취 -> 발화 종료 즉시 웹캠 캡처 -> VLM 추론
"""
import cv2
from config import settings
from server.repositories.connection import init_db_pool, close_db_pool
from server.services.audio_listener_service import audio_listener_service
from server.services.vision_service import VisionService
from server.workers.ai_worker import ai_worker

def test_voice_vision_pipeline():
    print("=" * 60)
    print(" [테스트] 자연스러운 음성-비전 인터랙션 검증")
    print("=" * 60)

    init_db_pool()
    vision_service = VisionService()

    # 1. 카메라 미리 열어두기
    cap = cv2.VideoCapture(settings.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[-] 카메라(Index: {settings.CAMERA_INDEX})를 열 수 없습니다.")
        close_db_pool()
        return

    try:
        # 카메라 초기화 및 잔여 프레임 정리
        for _ in range(5):
            cap.read()

        # 2. 말을 먼저 듣기
        print("\n[*] 마이크 음성 청취 대기 중... 지금 사물을 비추며 말씀하세요!")
        print("    (예: '멘티오 이거 봐봐', '멘티오 이거 뭐야?')")
        audio_data = audio_listener_service.listen_phrase()

        if audio_data is None:
            print("[-] 음성이 감지되지 않았습니다.")
            return

        # 3. 말끝이 감지된 바로 그 순간 최신 프레임 캡처!
        ret, current_frame = cap.read()
        if not ret or current_frame is None:
            print("[-] 프레임 캡처 실패")
            return

        print("[+] 발화 종료 감지! 그 순간의 카메라 프레임 캡처 완료.")

        # 어떤 화면이 캡처되었는지 1초간 미리보기
        cv2.imshow("Captured Frame (Trigger Point)", current_frame)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()

        # VLM용 480p PIL Image 변환
        pil_snapshot = vision_service.prepare_snapshot_for_vlm(current_frame)

        # 4. 워커 파이프라인 구동
        print("[*] 멀티모달 파이프라인(AIWorker) 구동 중...")
        action = ai_worker.process_voice_interaction(
            audio_data=audio_data,
            current_frame=pil_snapshot
        )

        if action:
            print("\n" + "=" * 40)
            print(f"[*] [최종 반응 결과]")
            print(f"    - 감정(Emotion) : {action.emotion}")
            print(f"    - 대사(Speech)  : {action.speech}")
            print(f"    - LED RGB       : {action.led_rgb}")
            print("=" * 40)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        close_db_pool()

if __name__ == "__main__":
    test_voice_vision_pipeline()