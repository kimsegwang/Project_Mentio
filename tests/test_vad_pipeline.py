from server.services.audio_listener_service import audio_listener_service
from server.services.stt_service import stt_service
from server.services.intent_service import intent_service

def main():
    print("\n" + "=" * 60)
    print("[*] VAD 리스너 대기 중... 편하게 말씀하세요.")
    print("=" * 60)

    audio_data = audio_listener_service.listen_phrase()
    if audio_data is None:
        print("[-] 캡처된 음성이 없습니다.")
        return

    transcribed_text = stt_service.transcribe(audio_data)
    trigger_type, needs_vision = intent_service.analyze_voice_intent(transcribed_text)

    print("\n" + "-" * 50)
    print(f"[STT 텍스트]      : {transcribed_text}")
    print(f"[트리거 타입]    : {trigger_type}")
    print(f"[스냅샷 동봉 여부]: {needs_vision}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()