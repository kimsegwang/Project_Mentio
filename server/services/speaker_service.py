"""
server/services/speaker_service.py
화자 검증(Speaker Verification) 서비스.

Resemblyzer(경량 CPU 추론 화자 임베딩 모델)로 발화 오디오를 임베딩 벡터로 변환하고,
사전 등록된 기준 화자(primary_user) 임베딩과의 코사인 유사도로 동일 화자 여부를 판정한다.
DB/하드웨어에 의존하지 않는 순수 도메인 서비스이며, 입력은 AudioSource(VAD)가 캡처한
16kHz float32 오디오 배열 또는 PCM16 바이트를 그대로 받는다.
"""
import logging
import os
import threading
from typing import Optional, Union

import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class SpeakerService:
    def __init__(
        self,
        reference_embedding_path: str = None,
        similarity_threshold: float = None,
    ):
        self.reference_embedding_path = (
            reference_embedding_path
            if reference_embedding_path is not None
            else settings.SPEAKER_REFERENCE_EMBEDDING_PATH
        )
        self.similarity_threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else settings.SPEAKER_VERIFICATION_THRESHOLD
        )
        self._encoder = None
        self._load_lock = threading.Lock()
        self._reference_embedding: Optional[np.ndarray] = None
        self._reference_loaded = False

    def _get_encoder(self):
        """Resemblyzer VoiceEncoder 지연 로딩 (최초 1회, Mutex로 중복 로딩 방지)"""
        if self._encoder is None:
            with self._load_lock:
                if self._encoder is None:
                    from resemblyzer import VoiceEncoder
                    logger.info("[SpeakerService] Resemblyzer VoiceEncoder 로딩 중 (CPU)...")
                    self._encoder = VoiceEncoder("cpu")
        return self._encoder

    def embed(self, audio: Union[np.ndarray, bytes], sample_rate: int = 16000) -> np.ndarray:
        """16kHz 오디오(bytes 또는 float32 배열)를 화자 임베딩 벡터로 변환한다."""
        from resemblyzer import preprocess_wav

        if isinstance(audio, bytes):
            audio = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

        wav = preprocess_wav(audio, source_sr=sample_rate)
        encoder = self._get_encoder()
        return encoder.embed_utterance(wav)

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """두 벡터 간 코사인 유사도. 영벡터가 섞여도 예외 없이 0.0을 반환한다."""
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    def _load_reference_embedding(self) -> Optional[np.ndarray]:
        """기준 화자 임베딩(.npy)을 지연 로딩해 캐시한다. 파일이 없으면 미등록 상태(None)."""
        if not self._reference_loaded:
            if self.reference_embedding_path and os.path.exists(self.reference_embedding_path):
                self._reference_embedding = np.load(self.reference_embedding_path)
            else:
                self._reference_embedding = None
            self._reference_loaded = True
        return self._reference_embedding

    def reload_reference_embedding(self) -> None:
        """enroll_speaker.py 등으로 기준 임베딩 파일이 갱신된 후 캐시를 무효화한다."""
        self._reference_loaded = False
        self._reference_embedding = None

    def is_enrolled(self) -> bool:
        """기준 화자 임베딩이 등록되어 있는지 여부."""
        return self._load_reference_embedding() is not None

    def verify(self, audio: Union[np.ndarray, bytes], sample_rate: int = 16000) -> bool:
        """
        VAD 직후 캡처된 발화 오디오가 기준 화자(primary_user)와 동일 인물인지 판정한다.

        - 비활성화 플래그(SPEAKER_VERIFICATION_ENABLED=False) 또는 기준 임베딩 미등록 시:
          검증을 스킵하고 True(통과)를 반환해 파이프라인이 끊기지 않도록 한다.
        - 임베딩 추출/비교 중 예외가 발생해도 안전하게 True로 폴백한다(가용성 우선).
        """
        if not settings.SPEAKER_VERIFICATION_ENABLED:
            return True

        reference = self._load_reference_embedding()
        if reference is None:
            logger.info("[SpeakerService] 기준 화자 임베딩 미등록 상태 -> 검증 스킵")
            return True

        try:
            candidate = self.embed(audio, sample_rate=sample_rate)
            similarity = self.cosine_similarity(candidate, reference)
            is_match = similarity >= self.similarity_threshold
            logger.info(
                f"[SpeakerService] 화자 검증 결과: {'통과' if is_match else '차단'} "
                f"(유사도: {similarity:.3f}, 임계값: {self.similarity_threshold})"
            )
            return is_match
        except Exception as e:
            logger.warning(f"[SpeakerService] 화자 검증 처리 중 예외 발생, 안전하게 통과 처리: {e}")
            return True


# Spring Bean 스타일 전역 싱글톤 등록
speaker_service = SpeakerService()
