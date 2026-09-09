"""
server/services/audio_listener_service.py
실시간 오디오 스트림에서 음성 에너지(RMS) 기반으로 발화 시작과 종료를 감지(VAD)하는 서비스.
"""
import collections
import logging
import time
from typing import Optional
import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


class AudioListenerService:
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 30,
        energy_threshold: float = 0.015,
        silence_timeout_sec: float = 0.8,
        pre_roll_sec: float = 0.3,
        max_record_sec: float = 10.0
    ):
        """
        Args:
            sample_rate: 샘플링 주파수 (16kHz 고정)
            chunk_duration_ms: 단일 청크 분석 단위 (30ms)
            energy_threshold: 발화 감지 RMS 임계값
            silence_timeout_sec: 발화 종료로 판단할 무음 지속 시간
            pre_roll_sec: 첫 단어 짤림 방지용 사전 버퍼 시간
            max_record_sec: 단일 발화 최대 허용 시간
        """
        self.sample_rate = sample_rate
        self.chunk_size = int(sample_rate * (chunk_duration_ms / 1000.0))
        self.energy_threshold = energy_threshold
        self.silence_timeout_sec = silence_timeout_sec
        self.max_record_sec = max_record_sec

        # 사전 롤 버퍼 (첫 음절 보존)
        pre_roll_chunks = int(pre_roll_sec / (chunk_duration_ms / 1000.0))
        self.pre_roll_buffer = collections.deque(maxlen=pre_roll_chunks)

        logger.info(
            f"AudioListenerService initialized: sample_rate={sample_rate}Hz, "
            f"threshold={energy_threshold}, silence_timeout={silence_timeout_sec}s"
        )

    def _calculate_rms(self, chunk: np.ndarray) -> float:
        """오디오 청크의 RMS(Root Mean Square) 에너지를 계산합니다."""
        return float(np.sqrt(np.mean(chunk**2)))

    def listen_phrase(self) -> Optional[np.ndarray]:
        """
        마이크 입력을 대기하다가 발화 시작부터 종료까지의 음성을 캡처해 반환합니다.

        Returns:
            Optional[np.ndarray]: 캡처된 16kHz float32 1D 오디오 배열 (미발화 시 None)
        """
        logger.info("Listening for speech...")
        self.pre_roll_buffer.clear()
        recorded_frames = []
        is_speaking = False
        silence_start_time = None
        recording_start_time = None

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.chunk_size
        ) as stream:
            while True:
                chunk, overflowed = stream.read(self.chunk_size)
                if overflowed:
                    logger.warning("Audio input buffer overflowed.")

                flat_chunk = chunk.flatten()
                rms = self._calculate_rms(flat_chunk)

                current_time = time.time()

                if not is_speaking:
                    self.pre_roll_buffer.append(flat_chunk)
                    # 발화 시작 판별
                    if rms > self.energy_threshold:
                        is_speaking = True
                        recording_start_time = current_time
                        logger.info(f"Speech detected (RMS: {rms:.4f}). Recording started.")
                        # 사전 버퍼 데이터 누적
                        recorded_frames.extend(list(self.pre_roll_buffer))
                        recorded_frames.append(flat_chunk)
                else:
                    recorded_frames.append(flat_chunk)

                    # 무음 구간 판별
                    if rms < self.energy_threshold:
                        if silence_start_time is None:
                            silence_start_time = current_time
                        elif current_time - silence_start_time >= self.silence_timeout_sec:
                            logger.info("Silence detected. Speech finished.")
                            break
                    else:
                        silence_start_time = None

                    # 최대 녹음 시간 초과 방어
                    if current_time - recording_start_time >= self.max_record_sec:
                        logger.info("Max record duration reached. Forcing cut.")
                        break

        if recorded_frames:
            full_audio = np.concatenate(recorded_frames, axis=0)
            return full_audio
        return None


# Spring Bean 스타일 싱글톤 등록
audio_listener_service = AudioListenerService()