"""
server/adapters/audio_io.py
오디오 입출력 하드웨어(PC 사운드카드, 향후 ESP32 WebSocket 스트림 등)를
서비스 계층으로부터 격리하기 위한 추상 인터페이스.

BrainService/IntentService/AIWorker 등 도메인 서비스는 이 인터페이스에만
의존해야 하며, 구체적인 입출력 구현(PC sounddevice/pygame, ESP32 WebSocket 등)에
직접 의존해서는 안 된다 (의존성 역전 원칙, DIP).
"""
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class AudioSource(ABC):
    """음성 입력원 인터페이스. 발화 시작~종료 구간을 캡처해 반환한다."""

    @abstractmethod
    def listen_phrase(self) -> Optional[np.ndarray]:
        """
        발화 시작부터 종료까지의 음성을 캡처해 반환한다.

        Returns:
            Optional[np.ndarray]: 캡처된 16kHz float32 1D 오디오 배열 (미발화 시 None)
        """
        raise NotImplementedError


class AudioSink(ABC):
    """음성 출력 대상 인터페이스. 합성된 오디오 바이트를 재생하고 완료까지 대기한다."""

    @abstractmethod
    def play_bytes_and_wait(self, audio_bytes: bytes, reverb_guard_sec: float = 0.0) -> None:
        """
        오디오 바이트 스트림을 재생하고, 재생 및 잔향 가드 대기까지 완료된 후 반환한다.

        Args:
            audio_bytes: 재생할 오디오 바이트 스트림
            reverb_guard_sec: 재생 종료 후 추가로 대기할 잔향 가드 시간(초)
        """
        raise NotImplementedError
