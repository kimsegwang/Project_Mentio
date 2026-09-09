"""
tests/test_vad_pipeline.py
VAD 리스너 -> STT -> 이중화(Regex + LLM) 의도 분석 E2E 검증.
"""
from server.services.audio_listener_service import audio_listener_service
from server.services.stt_service import stt_service
from server.services.intent_service import intent_service
from server.services.brain_service import brain_service

def main():
    print("\n" + "=" * 60)
    print("[*] VAD 리스너 대기 중... 편하게 말씀하세요.")
    print("    (말을 시작하면 감지하고, 말씀 후 0.8초 침묵하면 자동으로 분석합니다)")
    print("=" * 60)

    # 1. VAD 자동 캡처
    audio_data = audio_listener_service.listen_phrase()

    if audio_data is None or len(audio_data) == 0:
        print("[-] 캡처된 음성이 없습니다.")
        return

    duration = len(audio_data) / 16000.0
    print(f"\n[*] 발화 종료 감지! (총 {duration:.2f}초 분량)")
    print("[*] 텍스트 변환 및 이중화 의도 분석 진행 중...")

    # 2. STT 변환
    transcribed_text = stt_service.transcribe(audio_data)

    # 3. 이중화 의도 판단 (1차 정규식 통과 못하면 2차 Gemini 판별)
    trigger_type, needs_vision = intent_service.analyze_voice_intent(
        transcribed_text,
        llm_classifier=brain_service.classify_vision_intent
    )

    print("\n" + "-" * 50)
    print(f"[STT 텍스트]      : {transcribed_text}")
    print(f"[트리거 타입]    : {trigger_type}")
    print(f"[스냅샷 필요 여부]: {needs_vision}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()