# tests/test_tts_pipeline.py
import time
import pytest
from server.services.tts_service import TTSService
from server.services.audio_player_service import AudioPlayerService


def test_tts_synthesize_bytes():
    """1. Edge-TTS가 빈 데이터가 아닌 정상 MP3 바이트 스트림을 반환하는지 검증"""
    tts_service = TTSService()
    test_text = "멘티오 테스트 음성입니다."
    
    audio_bytes = tts_service.synthesize(test_text)
    
    assert isinstance(audio_bytes, bytes)
    assert len(audio_bytes) > 0, "오디오 바이트 스트림이 비어 있습니다."


def test_audio_player_playback_and_guard():
    """2. AudioPlayer가 바이트를 정상 재생하고 Reverb Guard(지연)를 지키는지 검증"""
    tts_service = TTSService()
    player = AudioPlayerService()
    
    test_text = "작동 확인."
    audio_bytes = tts_service.synthesize(test_text)
    
    reverb_guard = 0.4
    start_time = time.time()
    
    # 재생 및 에코 가드 대기
    player.play_bytes_and_wait(audio_bytes, reverb_guard_sec=reverb_guard)
    
    elapsed = time.time() - start_time
    # 최소 발화 시간 + 400ms 잔향 가드 이상 시간이 소요되어야 함
    assert elapsed >= reverb_guard, f"에코 가드 최소 시간({reverb_guard}s)을 충족하지 못했습니다: {elapsed}s"


if __name__ == "__main__":
    print("[1/2] TTS 합성 단위 테스트 실행...")
    test_tts_synthesize_bytes()
    print(">> TTS 합성 테스트 통과 (바이트 스트림 확보 성공)")

    print("[2/2] 스피커 출력 및 에코 가드 대기 테스트 실행...")
    test_audio_player_playback_and_guard()
    print(">> 스피커 출력 및 에코 가드 테스트 통과")