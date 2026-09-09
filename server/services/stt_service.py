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
    # 모델에 사전 주입할 고유명사 및 도메인 컨텍스트
    DOMAIN_INITIAL_PROMPT = (
        "안녕 멘티오, 멘티오야 오늘 날씨 어때? 주변 한번 봐봐. "
        "사진 찍어줘. 반가워, 지금 뭐 하고 있어? 반려로봇 대화."
    )

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 6
    ):
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
        initial_prompt: str = None
    ) -> str:
        """
        오디오 입력을 받아 한국어 텍스트로 변환합니다.
        """
        try:
            if isinstance(audio_input, bytes):
                audio_input = io.BytesIO(audio_input)

            prompt_text = initial_prompt if initial_prompt else self.DOMAIN_INITIAL_PROMPT

            segments, info = self.model.transcribe(
                audio_input,
                language="ko",
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                initial_prompt=prompt_text,
                beam_size=1,                        # 5 -> 1 로 변경 (속도 대폭 향상)
                best_of=1,                          #단일 추론 고정
                temperature=0.0,                    # 가장 정확한 토큰만 선택 (오타 방지)
                condition_on_previous_text=False,   # 이전 문맥 왜곡 방지
                no_speech_threshold=0.6,
                repetition_penalty=1.1              # 끝자리 반복 억제
            )

            transcribed_text = " ".join([segment.text.strip() for segment in segments]).strip()
            logger.info(f"STT Transcribed ({info.duration:.2f}s audio): '{transcribed_text}'")
            return transcribed_text

        except Exception as e:
            logger.error(f"Failed to transcribe audio: {e}", exc_info=True)
            return ""


# Spring Bean 스타일 전역 싱글톤 등록
stt_service = STTService()