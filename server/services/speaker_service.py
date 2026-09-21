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
from typing import List, Optional, Union

import numpy as np

from config import settings
from server.repositories.speaker_repository import get_active_speaker_embeddings
from server.schemas.speaker import SpeakerIdentificationResult, SpeakerVerificationResult

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
        # [다중 사용자 식별] 직전에 통과한 화자의 user_id/display_name. 세션 소프트패스로
        # 통과할 때 "같은 사람이 계속 말하는 중"이라는 가정 하에 동일 화자로 귀속시키는 데 쓰인다.
        self._last_passed_user_id: Optional[str] = None
        self._last_passed_display_name: Optional[str] = None
        # [다중 사용자 식별] DB(speaker_profiles)에서 조회한 활성 화자 임베딩 캐시.
        # enroll_speaker.py로 신규 등록/갱신 후에는 reload_speaker_profiles()로 무효화해야 한다.
        self._profiles_lock = threading.Lock()
        self._active_profiles: Optional[List] = None
        self._profiles_loaded = False

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
    def _coerce_to_float_array(value) -> np.ndarray:
        """
        임베딩 값을 1차원 float32 np.ndarray로 강제 변환한다.

        [pgvector 타입 방어] DB(speaker_profiles)에서 조회된 임베딩은 register_vector() 등록
        여부/pgvector-python 버전에 따라 순수 np.ndarray/list가 아닌 pgvector.Vector 객체로
        반환될 수 있다. Vector 객체를 dtype 지정 없이 np.asarray()에 바로 넘기면 원소별 변환에
        실패해(0차원 object 배열로 감싸지거나 TypeError 발생) 코사인 유사도 계산에서
        "unsupported operand type(s) for *: 'Vector' and 'Vector'" 같은 타입 에러로 이어졌다.
        pgvector.Vector가 제공하는 to_list()를 우선 시도해 list로 평탄화한 뒤 dtype=np.float32로
        변환하고, 1차원으로 reshape해 항상 순수 float 배열끼리 연산하도록 보장한다.
        """
        if hasattr(value, "to_list"):
            value = value.to_list()
        return np.asarray(value, dtype=np.float32).reshape(-1)

    @staticmethod
    def cosine_similarity(a: Union[np.ndarray, list], b: Union[np.ndarray, list]) -> float:
        """두 벡터 간 코사인 유사도. 영벡터가 섞여도 예외 없이 0.0을 반환한다."""
        a = SpeakerService._coerce_to_float_array(a)
        b = SpeakerService._coerce_to_float_array(b)
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

    def _load_active_profiles(self) -> List:
        """DB(speaker_profiles)의 활성 화자 임베딩을 지연 로딩해 캐시한다."""
        if not self._profiles_loaded:
            with self._profiles_lock:
                if not self._profiles_loaded:
                    self._active_profiles = get_active_speaker_embeddings()
                    self._profiles_loaded = True
        return self._active_profiles or []

    def reload_speaker_profiles(self) -> None:
        """enroll_speaker.py 등으로 화자 프로필이 신규 등록/갱신된 후 캐시를 무효화한다."""
        with self._profiles_lock:
            self._profiles_loaded = False
            self._active_profiles = None

    def has_enrolled_speakers(self) -> bool:
        """1:N 식별 대상으로 등록된 활성 화자가 한 명이라도 있는지 여부."""
        return len(self._load_active_profiles()) > 0

    def identify_speaker(
        self, audio: Union[np.ndarray, bytes], sample_rate: int = 16000
    ) -> SpeakerIdentificationResult:
        """
        VAD 직후 캡처된 발화 오디오와 가장 유사한 등록 화자를 1:N 코사인 유사도로 찾는다.

        verify()와 동일한 판정 가드(비활성화/미등록 시 스킵, 짧은 발화 완화 임계값, 세션
        소프트패스, 예외 시 가용성 우선 폴백)를 그대로 유지하되, 기준이 되는 단일 임베딩이
        아니라 DB에 등록된 모든 활성 화자 임베딩 중 최고 유사도 후보를 선택한다.

        [미등록 화자 정책] 등록된 화자가 있는 상태에서 어떤 후보와도 임계값 이상 일치하지
        않으면(세션 소프트패스도 아니면) is_match=False로 반환한다. 호출부(AIWorker)는 이를
        "제3자/미등록 화자의 발화"로 간주해 파이프라인 진입 자체를 차단(무시)해야 한다.
        검증이 비활성화되었거나 등록된 화자가 전무한 경우에는 skipped=True와 함께
        DEFAULT_USER_ID로 판정해, 화자 등록 전 상태의 기존 단일 사용자 동작을 그대로 보존한다.
        """
        if not settings.SPEAKER_VERIFICATION_ENABLED:
            return SpeakerIdentificationResult(is_match=True, skipped=True, user_id=settings.DEFAULT_USER_ID)

        profiles = self._load_active_profiles()
        if not profiles:
            logger.info("[SpeakerService] 등록된 화자 프로필 없음 -> 식별 스킵")
            return SpeakerIdentificationResult(is_match=True, skipped=True, user_id=settings.DEFAULT_USER_ID)

        duration_sec = self._audio_duration_sec(audio, sample_rate)
        is_short_utterance = duration_sec <= settings.SPEAKER_SHORT_UTTERANCE_MAX_SEC
        threshold = (
            settings.SPEAKER_SHORT_UTTERANCE_THRESHOLD
            if is_short_utterance
            else self.similarity_threshold
        )

        try:
            candidate = self.embed(audio, sample_rate=sample_rate)

            best_profile = None
            best_similarity = -1.0
            for profile in profiles:
                # cosine_similarity 내부에서 dtype=np.float32 강제 변환을 수행하므로,
                # profile.embedding이 list/np.ndarray/pgvector.Vector 무엇이든 안전하게 처리된다.
                similarity = self.cosine_similarity(candidate, profile.embedding)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_profile = profile

            is_match = best_similarity >= threshold
            soft_passed = False
            result_user_id = best_profile.user_id if best_profile is not None else None
            result_display_name = best_profile.display_name if best_profile is not None else None

            if not is_match and self._is_within_session_soft_pass_window():
                is_match = True
                soft_passed = True
                # 세션이 이어지는 중이라는 가정 하에, 이번 최고 유사도 후보가 아니라 직전에
                # 실제로 통과했던 화자로 귀속시킨다 (짧은 맞장구 등은 후보 매칭이 부정확할 수 있음).
                result_user_id = self._last_passed_user_id or result_user_id
                result_display_name = self._last_passed_display_name or result_display_name

            if is_match:
                self._last_passed_monotonic = time.monotonic()
                self._last_passed_user_id = result_user_id
                self._last_passed_display_name = result_display_name

            note = ""
            if soft_passed:
                note = f" [세션 소프트패스: 최근 {settings.SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC:.0f}초 이내 통과 이력]"
            elif is_short_utterance:
                note = f" [짧은 발화 {duration_sec:.2f}s, 완화 임계값 적용]"

            logger.info(
                f"[SpeakerService] 화자 식별 점수: {best_similarity:.3f} (기준: {threshold:.3f}) -> "
                f"{'통과 (user: ' + str(result_user_id) + ')' if is_match else '실패 (미등록 화자로 판단, 차단)'}{note}"
            )

            return SpeakerIdentificationResult(
                user_id=result_user_id if is_match else None,
                display_name=result_display_name if is_match else None,
                is_match=is_match,
                similarity=best_similarity,
                threshold=threshold,
                soft_passed=soft_passed,
            )
        except Exception as e:
            logger.warning(f"[SpeakerService] 화자 식별 처리 중 예외 발생, 안전하게 통과 처리: {e}")
            return SpeakerIdentificationResult(is_match=True, skipped=True, user_id=settings.DEFAULT_USER_ID)


# Spring Bean 스타일 전역 싱글톤 등록
speaker_service = SpeakerService()
