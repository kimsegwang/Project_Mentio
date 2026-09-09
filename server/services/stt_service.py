"""
server/services/stt_service.py
음성 데이터를 텍스트로 변환(faster-whisper)하는 단일 책임을 가진 서비스 계층.
"""
import io
import logging
from typing import Union
import numpy as np
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)


class STTService:
    def __init__(
        self,
        model_size: str = "small",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 4
    ):
        """
        Spring의 @PostConstruct 역할: 모델을 메모리에 1회 싱글톤 로드합니다.
        Ryzen 7 5700U 최적화: CPU INT8 양자화 및 4스레드 할당.
        """
        logger.info(
            f"Initializing Faster-Whisper: model={model_size}, device={device}, "
            f"compute_type={compute_type}, threads={cpu_threads}"
        )
        self.model = WhisperModel(
            model_size_or_path=model_size,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            download_root=None
        )
        logger.info("Faster-Whisper model loaded successfully.")

    def transcribe(
        self, 
        audio_input: Union[np.ndarray, bytes, str],
        initial_prompt: str = "멘티오, 탁상형 반려로봇과의 대화."
    ) -> str:
        """
        오디오 입력을 받아 한국어 텍스트로 변환합니다.

        Args:
            audio_input: 
                - np.ndarray: 16kHz float32 1D numpy array
                - bytes: WAV/PCM 파일 바이너리 버퍼
                - str: 로컬 오디오 파일 경로
            initial_prompt: 고유명사 및 도메인 인식 힌트
            
        Returns:
            str: 변환된 문자열 (공백 정리 완료)
        """
        try:
            if isinstance(audio_input, bytes):
                audio_input = io.BytesIO(audio_input)

            segments, info = self.model.transcribe(
                audio_input,
                language="ko",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=initial_prompt,
                beam_size=5
            )

            transcribed_text = " ".join([segment.text.strip() for segment in segments]).strip()
            logger.info(f"STT Transcribed ({info.duration:.2f}s audio): '{transcribed_text}'")
            return transcribed_text

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}", exc_info=True)
            return ""


# Spring Bean 스타일 전역 싱글톤 등록
stt_service = STTService()