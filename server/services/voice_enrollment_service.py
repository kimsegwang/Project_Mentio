"""
server/services/voice_enrollment_service.py
대화형 음성 온보딩(Conversational Voice Enrollment) 서비스.

터미널 스크립트(scripts/enroll_speaker.py) 없이 음성 대화만으로 신규 화자를 등록하는
멀티턴 상태 머신을 관리한다.

    IDLE --(등록 요청 발화)--> WAITING_FOR_NAME --(이름 응답)--> WAITING_FOR_VOICE_SAMPLE
         <--(샘플 수집 -> 임베딩 추출 -> speaker_profiles UPSERT -> 캐시 무효화: 등록 완료)--

- 등록 요청/취소/이름 추출은 모두 LLM 호출 없는 정규식 룰로 처리해 지연을 추가하지 않는다.
- 온보딩 진행 중(is_active)에는 AIWorker가 화자 식별 차단(Fail-Close)을 바이패스한다.
  대신 SPEAKER_ENROLLMENT_SESSION_TIMEOUT_SEC 동안 응답이 없으면 세션을 자동 만료시켜,
  바이패스 창이 무기한 열려 있지 않도록 한다.
- 오디오 캡처/TTS 재생 등 I/O는 전혀 다루지 않는 순수 도메인 서비스이며, 오디오 배열과
  STT 텍스트를 입력받아 안내 멘트(EnrollmentReply)만 반환한다.
"""
import logging
import re
import threading
import time
from datetime import datetime
from typing import Callable, Optional, Union

import numpy as np

from config import settings
from server.repositories.speaker_repository import upsert_speaker_profile
from server.schemas.action import EmotionType
from server.schemas.speaker import EnrollmentReply, EnrollmentState
from server.services.intent_service import IntentService
from server.services.speaker_service import SpeakerService, speaker_service

logger = logging.getLogger(__name__)


class VoiceEnrollmentService:
    # 등록 요청 패턴 ("목소리 등록할래", "내 목소리 기억해줘", "새 화자 등록")
    ENROLLMENT_REQUEST_PATTERNS = [
        r"(목소리|음성)\s*(좀\s*)?(등록|기억|저장|외워|익혀)",
        r"(새|신규|새로운)\s*(화자|사용자|멤버|가족)\s*(좀\s*)?(등록|추가)",
        r"화자\s*(좀\s*)?(등록|추가)",
    ]

    # 등록 요청 부정/회상 질의 ("등록하지 마", "내 목소리 기억나?")는 요청으로 보지 않는다
    ENROLLMENT_VETO_PATTERNS = [
        r"(등록|기억|저장|추가)\s*(하지|안\s*해|말아|취소)",
        r"기억\s*(나|하니|하냐|하고\s*있)",
    ]

    # 온보딩 취소 패턴 - 목소리 샘플 문장("이제 그만 자야지") 오탐을 막기 위해 발화 전체가
    # 취소 표현일 때만 인정한다.
    CANCEL_PATTERN = r"^(등록\s*)?(취소|그만|그만해|됐어|안\s*할래|안\s*해|하지\s*마)(\s*(할래|할게|해|해줘|요))?$"

    # 이름 응답에서 걷어낼 앞/뒤 군더더기 ("저는 민수예요", "첫째라고 불러줘")
    NAME_PREFIX_PATTERN = r"^((음+|어+|아+)\s+)?((제|내|저의|나의)\s*)?((이름|호칭)\s*(은|는)\s*)?((저|나)\s*는\s*|난\s+|전\s+)?"
    NAME_SUFFIX_PATTERNS = [
        r"\s*이?라고\s*(해|해요|합니다|불러.*)?$",
        r"\s*(이에요|이예요|예요|에요|입니다|이야|이요|야|요|임)$",
        r"\s*님$",
    ]
    MAX_NAME_LENGTH = 10

    PROMPT_ASK_NAME = "반가워요! 등록하실 분의 성함이나 호칭을 말씀해 주세요."
    PROMPT_RETRY_NAME = "잘 못 들었어요. 등록하실 분의 성함이나 호칭만 짧게 다시 말씀해 주세요."
    PROMPT_ASK_SAMPLE = "{name} 님 반가워요! '오늘 날씨가 참 좋다'처럼 평소 말투로 한 문장 말씀해 주세요."
    PROMPT_RETRY_SAMPLE = "목소리가 조금 짧았어요. '오늘 날씨가 참 좋다'처럼 한 문장을 조금 길게 말씀해 주세요."
    PROMPT_COMPLETED = "등록이 완료되었어요! {name} 님, 이제부터 목소리로 바로 알아볼게요."
    PROMPT_FAILED = "앗, 등록 중에 문제가 생겼어요. 잠시 후에 다시 시도해 주세요."
    PROMPT_CANCELLED = "알겠어요, 목소리 등록을 취소했어요."

    def __init__(
        self,
        speaker_service_instance: SpeakerService = speaker_service,
        session_timeout_sec: Optional[float] = None,
        min_sample_sec: Optional[float] = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.speaker_service = speaker_service_instance
        self.session_timeout_sec = (
            session_timeout_sec
            if session_timeout_sec is not None
            else settings.SPEAKER_ENROLLMENT_SESSION_TIMEOUT_SEC
        )
        self.min_sample_sec = (
            min_sample_sec if min_sample_sec is not None else settings.SPEAKER_ENROLLMENT_MIN_SAMPLE_SEC
        )
        self._clock = clock

        self.wake_regex = re.compile(IntentService.WAKE_WORDS_PATTERN)
        self.request_patterns = [re.compile(p) for p in self.ENROLLMENT_REQUEST_PATTERNS]
        self.veto_patterns = [re.compile(p) for p in self.ENROLLMENT_VETO_PATTERNS]
        self.cancel_regex = re.compile(self.CANCEL_PATTERN)
        self.name_prefix_regex = re.compile(self.NAME_PREFIX_PATTERN)
        self.name_suffix_patterns = [re.compile(p) for p in self.NAME_SUFFIX_PATTERNS]

        # 상태 전이는 음성 처리 스레드에서 일어나지만, 조회(is_active)는 다른 스레드에서도
        # 가능하므로 Mutex로 상태/대기 이름/마지막 활동 시각을 함께 보호한다.
        self._lock = threading.Lock()
        self._state = EnrollmentState.IDLE
        self._pending_name: Optional[str] = None
        self._last_activity: Optional[float] = None

    # --- 상태 조회 ---

    def _expire_if_timed_out_locked(self) -> None:
        if self._state == EnrollmentState.IDLE or self._last_activity is None:
            return
        if self._clock() - self._last_activity > self.session_timeout_sec:
            logger.info(
                f"[VoiceEnrollment] {self.session_timeout_sec:.0f}초 동안 응답이 없어 온보딩 세션 만료 "
                f"({self._state.value} -> IDLE)"
            )
            self._reset_locked()

    def _reset_locked(self) -> None:
        self._state = EnrollmentState.IDLE
        self._pending_name = None
        self._last_activity = None

    def _transition_locked(self, state: EnrollmentState) -> None:
        self._state = state
        self._last_activity = self._clock()

    @property
    def state(self) -> EnrollmentState:
        with self._lock:
            self._expire_if_timed_out_locked()
            return self._state

    def is_active(self) -> bool:
        """온보딩이 진행 중(이름/샘플 대기)인지 여부. True면 화자 식별 차단을 바이패스한다."""
        return self.state != EnrollmentState.IDLE

    # --- 텍스트 룰 ---

    def _normalize(self, text: str) -> str:
        clean_text = re.sub(r"[^\w\s]", " ", (text or "").strip())
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        target_text = self.wake_regex.sub("", clean_text).strip()
        return target_text or clean_text

    def is_enrollment_request(self, text: str) -> bool:
        """등록 요청 발화("목소리 등록할래", "내 목소리 기억해줘", "새 화자 등록") 여부."""
        target_text = self._normalize(text)
        if not target_text:
            return False
        if any(p.search(target_text) for p in self.veto_patterns):
            return False
        return any(p.search(target_text) for p in self.request_patterns)

    def is_cancel_request(self, text: str) -> bool:
        return bool(self.cancel_regex.match(self._normalize(text)))

    def extract_name(self, text: str) -> Optional[str]:
        """이름 응답 발화에서 순수 이름/호칭만 추출한다. 추출 불가(빈 값/너무 긴 문장)면 None."""
        name = self._normalize(text)
        name = self.name_prefix_regex.sub("", name).strip()
        for pattern in self.name_suffix_patterns:
            name = pattern.sub("", name).strip()
        if not name or len(name) > self.MAX_NAME_LENGTH:
            return None
        return name

    # --- 상태 전이 ---

    def start(self) -> EnrollmentReply:
        """등록 요청 감지 시 호출: IDLE(또는 진행 중이던 세션)을 WAITING_FOR_NAME으로 (재)시작한다."""
        with self._lock:
            self._pending_name = None
            self._transition_locked(EnrollmentState.WAITING_FOR_NAME)
        logger.info("[VoiceEnrollment] 온보딩 시작 -> WAITING_FOR_NAME")
        return EnrollmentReply(
            speech=self.PROMPT_ASK_NAME, emotion=EmotionType.HAPPY, state=EnrollmentState.WAITING_FOR_NAME
        )

    def cancel(self) -> EnrollmentReply:
        with self._lock:
            self._reset_locked()
        logger.info("[VoiceEnrollment] 사용자 요청으로 온보딩 취소 -> IDLE")
        return EnrollmentReply(
            speech=self.PROMPT_CANCELLED, emotion=EmotionType.NEUTRAL, state=EnrollmentState.IDLE
        )

    def handle_turn(
        self, audio: Union[np.ndarray, bytes], text: Optional[str], sample_rate: int = 16000
    ) -> Optional[EnrollmentReply]:
        """
        온보딩 진행 중 들어온 발화 한 턴을 처리한다. 진행 중인 세션이 없으면(만료 포함) None.
        - WAITING_FOR_NAME: STT 텍스트에서 이름을 추출해 WAITING_FOR_VOICE_SAMPLE로 전이.
        - WAITING_FOR_VOICE_SAMPLE: 오디오에서 화자 임베딩을 추출해 DB 적재 후 IDLE 복귀.
        두 단계 모두 발화 전체가 취소 표현이면 즉시 IDLE로 복귀한다.
        """
        with self._lock:
            self._expire_if_timed_out_locked()
            state = self._state
            pending_name = self._pending_name

        if state == EnrollmentState.IDLE:
            return None

        if text and self.is_cancel_request(text):
            return self.cancel()

        if state == EnrollmentState.WAITING_FOR_NAME:
            return self._handle_name(text)
        return self._handle_voice_sample(audio, pending_name, sample_rate)

    def _handle_name(self, text: Optional[str]) -> EnrollmentReply:
        name = self.extract_name(text or "")
        with self._lock:
            if name is None:
                # 재질문도 유효한 응답 대기이므로 타임아웃 기준 시각을 갱신한다
                self._transition_locked(EnrollmentState.WAITING_FOR_NAME)
            else:
                self._pending_name = name
                self._transition_locked(EnrollmentState.WAITING_FOR_VOICE_SAMPLE)

        if name is None:
            logger.info(f"[VoiceEnrollment] 이름 추출 실패 (발화: '{text}') -> 재질문")
            return EnrollmentReply(
                speech=self.PROMPT_RETRY_NAME, emotion=EmotionType.CURIOUS, state=EnrollmentState.WAITING_FOR_NAME
            )

        logger.info(f"[VoiceEnrollment] 이름 확인: '{name}' -> WAITING_FOR_VOICE_SAMPLE")
        return EnrollmentReply(
            speech=self.PROMPT_ASK_SAMPLE.format(name=name),
            emotion=EmotionType.HAPPY,
            state=EnrollmentState.WAITING_FOR_VOICE_SAMPLE,
        )

    def _handle_voice_sample(
        self, audio: Union[np.ndarray, bytes], name: str, sample_rate: int
    ) -> EnrollmentReply:
        duration_sec = self.speaker_service.audio_duration_sec(audio, sample_rate)
        if duration_sec < self.min_sample_sec:
            logger.info(
                f"[VoiceEnrollment] 샘플이 너무 짧음 ({duration_sec:.2f}s < {self.min_sample_sec:.2f}s) -> 재요청"
            )
            with self._lock:
                self._transition_locked(EnrollmentState.WAITING_FOR_VOICE_SAMPLE)
            return EnrollmentReply(
                speech=self.PROMPT_RETRY_SAMPLE,
                emotion=EmotionType.CURIOUS,
                state=EnrollmentState.WAITING_FOR_VOICE_SAMPLE,
            )

        user_id = self._generate_user_id()
        try:
            embedding = self.speaker_service.embed(audio, sample_rate=sample_rate)
            embedding_list = embedding.tolist() if hasattr(embedding, "tolist") else list(embedding)
            success = upsert_speaker_profile(user_id=user_id, display_name=name, embedding=embedding_list)
        except Exception as e:
            logger.warning(f"[VoiceEnrollment] 화자 임베딩 추출/적재 중 예외 발생: {e}")
            success = False

        with self._lock:
            self._reset_locked()

        if not success:
            logger.warning(f"[VoiceEnrollment] 화자 등록 실패 (name: {name}) -> IDLE")
            return EnrollmentReply(speech=self.PROMPT_FAILED, emotion=EmotionType.SAD, state=EnrollmentState.IDLE)

        # [Hot Reload] 캐시를 즉시 무효화해 서버 재기동 없이 다음 턴부터 새 화자로 식별되도록 하고,
        # 방금 목소리를 들려준 신규 화자를 세션 소프트패스 기준으로 지정한다.
        self.speaker_service.reload_speaker_profiles()
        self.speaker_service.anchor_session_speaker(user_id, name)
        logger.info(f"[VoiceEnrollment] 화자 등록 완료 (user: {user_id}, name: {name}, 샘플: {duration_sec:.2f}s)")

        return EnrollmentReply(
            speech=self.PROMPT_COMPLETED.format(name=name),
            emotion=EmotionType.HAPPY,
            state=EnrollmentState.IDLE,
            completed=True,
            user_id=user_id,
            display_name=name,
        )

    @staticmethod
    def _generate_user_id() -> str:
        return f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"


# Spring Bean 스타일 전역 싱글톤 등록
voice_enrollment_service = VoiceEnrollmentService()
