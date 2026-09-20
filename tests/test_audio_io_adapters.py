"""
tests/test_audio_io_adapters.py
오디오 입출력 인터페이스(AudioSource/AudioSink) 계층 분리 검증.

PC 사운드(sounddevice/pygame) 구현체를 실제로 구동하지 않고도,
- 기존 구현체들이 인터페이스를 올바르게 만족하는지
- AIWorker 등 상위 도메인 로직이 구체 클래스가 아닌 인터페이스에만 의존해
  가짜(Fake) 구현체 주입만으로 하드웨어 없이 동작하는지
를 검증한다. 이는 "PC 사운드 -> ESP32 WebSocket 스트리밍 교체 시 핵심 비즈니스
로직 0줄 수정" 원칙이 실제로 지켜지고 있음을 보장하는 회귀 테스트다.
"""
import numpy as np
import pytest

from server.adapters.audio_io import AudioSource, AudioSink
from server.services.audio_listener_service import AudioListenerService
from server.services.audio_player_service import AudioPlayerService
from server.workers.ai_worker import AIWorker


class FakeAudioSource(AudioSource):
    """하드웨어 없이 고정된 오디오 프레임을 반환하는 테스트용 더미 입력원."""

    def __init__(self, frame: np.ndarray = None):
        self.frame = frame if frame is not None else np.zeros(16000, dtype=np.float32)
        self.call_count = 0

    def listen_phrase(self):
        self.call_count += 1
        return self.frame


class FakeAudioSink(AudioSink):
    """하드웨어 없이 재생 호출만 기록하는 테스트용 더미 출력 대상."""

    def __init__(self):
        self.played = []

    def play_bytes_and_wait(self, audio_bytes: bytes, reverb_guard_sec: float = 0.0) -> None:
        self.played.append((audio_bytes, reverb_guard_sec))


def test_audio_listener_service_implements_audio_source():
    """기존 PC 구현체가 AudioSource 인터페이스를 실제로 구현하는지 검증."""
    listener = AudioListenerService()
    assert isinstance(listener, AudioSource)


def test_audio_player_service_implements_audio_sink():
    """기존 PC 구현체가 AudioSink 인터페이스를 실제로 구현하는지 검증."""
    player = AudioPlayerService()
    assert isinstance(player, AudioSink)


def test_audio_source_is_abstract():
    """AudioSource는 listen_phrase 미구현 시 직접 인스턴스화할 수 없어야 한다."""
    with pytest.raises(TypeError):
        AudioSource()


def test_audio_sink_is_abstract():
    """AudioSink는 play_bytes_and_wait 미구현 시 직접 인스턴스화할 수 없어야 한다."""
    with pytest.raises(TypeError):
        AudioSink()


def test_fake_audio_source_satisfies_interface():
    """가짜 구현체도 인터페이스를 상속하면 동일하게 취급 가능해야 한다 (구조적 대체 가능성)."""
    fake = FakeAudioSource()
    assert isinstance(fake, AudioSource)
    audio = fake.listen_phrase()
    assert isinstance(audio, np.ndarray)


class StubTTSService:
    """네트워크(edge-tts) 호출 없이 고정 바이트를 반환하는 테스트용 TTS 대역."""

    def synthesize(self, text: str) -> bytes:
        return b"FAKE_AUDIO_BYTES"


def test_ai_worker_accepts_fake_audio_sink_without_hardware():
    """
    AIWorker(상위 도메인 로직)가 구체 PC 구현체(pygame) 없이도
    AudioSink 인터페이스를 만족하는 가짜 구현체 주입만으로 동작해야 한다.
    이는 하드웨어 계층 교체 시 AIWorker 코드 수정이 0줄이어야 함을 보증한다.
    (TTS 자체는 네트워크(edge-tts)에 의존하므로 대역 객체로 분리해 순수 오디오
    출력 경로만 검증한다.)
    """
    fake_sink = FakeAudioSink()
    worker = AIWorker(
        tts_service_instance=StubTTSService(),
        audio_player_instance=fake_sink,
    )

    assert worker.audio_player is fake_sink

    worker._play_speech_and_guard("테스트 발화")

    assert len(fake_sink.played) == 1
    played_bytes, _reverb_guard = fake_sink.played[0]
    assert played_bytes == b"FAKE_AUDIO_BYTES"
