"""
server/services/speaker_service.py
화자 검증(Speaker Verification) 서비스.

Resemblyzer(경량 CPU 추론 화자 임베딩 모델)로 발화 오디오를 임베딩 벡터로 변환하고,
사전 등록된 기준 화자(primary_user) 임베딩과의 코사인 유사도로 동일 화자 여부를 판정한다.
DB/하드웨어에 의존하지 않는 순수 도메인 서비스이며, 입력은 AudioSource(VAD)가 캡처한
16kHz float32 오디오 배열 또는 PCM16 바이트를 그대로 받는다.

[실기 튜닝 노트] 최초 배포(SPEAKER_VERIFICATION_THRESHOLD=0.75) 후 실기 테스트에서 본인
목소리임에도 발화의 60% 이상이 "화자 검증 실패"로 차단되는 현상이 확인되었다. 원인 조사 결과:
  1) 판정 로그가 logger.info로만 남아 있었는데 루트 로거에 핸들러가 구성되어 있지 않아
     (Python 기본 WARNING 레벨) 실측 유사도 점수가 콘솔에 전혀 보이지 않아 원인 파악이 늦어짐.
  2) 임계값(0.75)이 Resemblyzer 실측 동일 화자 유사도 분포(0.65~0.74 다수 분포)에 비해
     지나치게 높게 설정되어 있었음.
  3) 1초 미만의 짧은 발화("네", "응" 등)는 Resemblyzer가 화자 특징을 충분히 추출하지 못해
     동일 화자라도 유사도 점수가 급락하는 경향이 실기 로그로 확인됨.
이에 따라 (1) 판정 결과를 항상 콘솔에 출력하도록 로거를 명시적으로 구성하고, (2) 기본
임계값을 0.68로 하향하고, (3) 짧은 발화 완화 임계값 및 대화 세션 소프트패스를 추가했다.
"""
import logging
import os
import threading
import time
from typing import Optional, Union

import numpy as np

from config import settings
from server.schemas.speaker import SpeakerVerificationResult

logger = logging.getLogger(__name__)
if not logger.handlers:
    # 루트 로거에 핸들러/레벨이 구성되어 있지 않으면(기본 WARNING) 화자 검증 점수 로그가
    # 콘솔에 전혀 출력되지 않아 임계값 튜닝 자체가 불가능해지므로, memory_service.py와
    # 동일하게 이 모듈 전용 콘솔 핸들러를 직접 구성한다.
    _console_handler = logging.StreamHandler()
    _console_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_console_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


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
        # [세션 소프트패스] 직전에 검증을 통과(하드/소프트 무관)한 시각(monotonic).
        # 단일 오디오 파이프라인 스레드에서만 갱신/조회되므로 별도 락 없이 안전하다.
        self._last_passed_monotonic: Optional[float] = None

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

    @staticmethod
    def _audio_duration_sec(audio: Union[np.ndarray, bytes], sample_rate: int) -> float:
        """검증 대상 오디오의 길이(초)를 계산한다. 짧은 발화 완화 임계값 판단에 사용."""
        if sample_rate <= 0:
            return 0.0
        num_samples = (len(audio) // 2) if isinstance(audio, bytes) else len(audio)
        return num_samples / sample_rate

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

    def _is_within_session_soft_pass_window(self) -> bool:
        """직전에 검증을 통과(하드/소프트 무관)한 지 SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC
        이내인, "대화 세션이 이어지는 중"인 상태인지 여부."""
        if self._last_passed_monotonic is None:
            return False
        elapsed = time.monotonic() - self._last_passed_monotonic
        return elapsed <= settings.SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC

    def verify(self, audio: Union[np.ndarray, bytes], sample_rate: int = 16000) -> SpeakerVerificationResult:
        """
        VAD 직후 캡처된 발화 오디오가 기준 화자(primary_user)와 동일 인물인지 판정한다.

        - 비활성화 플래그(SPEAKER_VERIFICATION_ENABLED=False) 또는 기준 임베딩 미등록 시:
          검증을 스킵하고 통과(is_match=True, skipped=True)를 반환해 파이프라인이 끊기지
          않도록 한다.
        - 오디오 길이가 SPEAKER_SHORT_UTTERANCE_MAX_SEC 이하인 짧은 발화는 Resemblyzer
          특징량 부족으로 점수가 급락하는 경향이 있어 완화된 SPEAKER_SHORT_UTTERANCE_THRESHOLD를
          적용한다.
        - 임계값 미달이더라도 직전 통과로부터 SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC 이내라면
          같은 대화 세션이 이어지는 것으로 간주해 소프트패스로 통과시킨다.
        - 임베딩 추출/비교 중 예외가 발생해도 안전하게 통과로 폴백한다(가용성 우선).
        - 계산된 유사도 점수와 적용 임계값은 항상 콘솔에 로깅한다(임계값 튜닝 가시성 확보).
        """
        if not settings.SPEAKER_VERIFICATION_ENABLED:
            return SpeakerVerificationResult(is_match=True, skipped=True)

        reference = self._load_reference_embedding()
        if reference is None:
            logger.info("[SpeakerService] 기준 화자 임베딩 미등록 상태 -> 검증 스킵")
            return SpeakerVerificationResult(is_match=True, skipped=True)

        duration_sec = self._audio_duration_sec(audio, sample_rate)
        is_short_utterance = duration_sec <= settings.SPEAKER_SHORT_UTTERANCE_MAX_SEC
        threshold = (
            settings.SPEAKER_SHORT_UTTERANCE_THRESHOLD
            if is_short_utterance
            else self.similarity_threshold
        )

        try:
            candidate = self.embed(audio, sample_rate=sample_rate)
            similarity = self.cosine_similarity(candidate, reference)
            is_match = similarity >= threshold
            soft_passed = False

            if not is_match and self._is_within_session_soft_pass_window():
                is_match = True
                soft_passed = True

            if is_match:
                self._last_passed_monotonic = time.monotonic()

            note = ""
            if soft_passed:
                note = f" [세션 소프트패스: 최근 {settings.SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC:.0f}초 이내 통과 이력]"
            elif is_short_utterance:
                note = f" [짧은 발화 {duration_sec:.2f}s, 완화 임계값 적용]"

            logger.info(
                f"[SpeakerService] 화자 검증 점수: {similarity:.3f} (기준: {threshold:.3f}) -> "
                f"{'통과' if is_match else '실패 (차단)'}{note}"
            )

            return SpeakerVerificationResult(
                is_match=is_match,
                similarity=similarity,
                threshold=threshold,
                soft_passed=soft_passed,
            )
        except Exception as e:
            logger.warning(f"[SpeakerService] 화자 검증 처리 중 예외 발생, 안전하게 통과 처리: {e}")
            return SpeakerVerificationResult(is_match=True, skipped=True)


# Spring Bean 스타일 전역 싱글톤 등록
speaker_service = SpeakerService()
