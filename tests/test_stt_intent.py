"""
test_stt_intent.py
stt_service와 intent_service의 계층 분리 협업 동작을 테스트합니다.
"""
import sounddevice as sd
from server.services.stt_service import stt_service
from server.services.intent_service import intent_service

SAMPLE_RATE = 16000
DURATION = 3

print(f"[*] 3초간 마이크로 말씀해 주세요 (예: '멘티오 내 표정 어때 보여?' 또는 '오늘 날씨 어때?')...")
audio = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
sd.wait()
print("[*] 녹음 완료, 계층 파이프라인 처리 중...")

# 1. STT Service: 음성 -> 텍스트
text = stt_service.transcribe(audio.flatten())

# 2. Intent Service: 텍스트 -> 의도 및 시각 필요 여부 판별
trigger_type, needs_vision = intent_service.analyze_voice_intent(text)

print("\n" + "=" * 50)
print(f"[STT Service 결과]    : {text}")
print(f"[Intent Service 트리거]: {trigger_type}")
print(f"[스냅샷 필요 여부]      : {needs_vision}")
print("=" * 50)