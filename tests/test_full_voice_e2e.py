"""
tests/test_full_voice_e2e.py
마이크 발화부터 AIWorker의 RobotAction 도출까지 전 구간 E2E 파이프라인 검증.
"""
from server.services.audio_listener_service import audio_listener_service
from server.workers.ai_worker import ai_worker

def main():
    print("\n" + "=" * 65)
    print("[*] 멘티오 음성 대화 시스템 가동 중... 편하게 말을 건네보세요.")
    print("    (예: '멘티오 오늘 날씨 어때?', '멘티오 안녕 반가워')")
    print("=" * 65)

    # 1. 음성 자동 캡처 (VAD)
    audio_data = audio_listener_service.listen_phrase()
    if audio_data is None:
        print("[-] 입력된 음성이 없습니다.")
        return

    print("\n[*] 발화 종료 감지! AIWorker 전체 파이프라인 구동 중...")

    # 2. AIWorker 단일 호출 파이프라인 실행
    action = ai_worker.process_voice_interaction(audio_data)

    if action:
        print("\n" + "=" * 50)
        print(f"🤖 [멘티오 감정 상태] : {action.emotion}")
        print(f"💡 [LED 색상 (RGB)]  : {action.led_rgb}")
        print(f"💬 [멘티오 발화 대사] : {action.speech}")
        print(f"⏱️  [동작 유지 시간]   : {action.duration}초")
        print("=" * 50 + "\n")
    else:
        print("[-] 응답 생성 실패")

if __name__ == "__main__":
    main()