"""
server/services/voice_enrollment_service.py
대화형 음성 온보딩(Conversational Voice Enrollment) 서비스.

터미널 스크립트(scripts/enroll_speaker.py) 없이 음성 대화만으로 신규 화자를 등록하는
멀티턴 상태 머신을 관리한다.

    IDLE --(등록 요청 발화)--> WAITING_FOR_NAME --(이름 응답)--> WAITING_FOR_VOICE_SAMPLE x N
         <--(샘플 N개 수집 -> 평균 임베딩 -> 중복 검사 -> speaker_profiles UPSERT -> 캐시 무효화)--

- [다중 발화 평균화] 이름 발화와 샘플 발화 N개(SPEAKER_ENROLLMENT_REQUIRED_SAMPLES)의 임베딩을
  평균 낸 뒤 L2 정규화해 최종 기준 벡터로 등록한다. 단일 발화 기준 벡터보다 톤/거리 변화에 강건하다.
- [중복 등록 방지] 등록 직전 기존 활성 화자와의 최고 유사도로 분기한다.
    >= SPEAKER_ENROLLMENT_DUPLICATE_THRESHOLD        : 기존 사용자 재등록 -> 기존 프로필 목소리 갱신
    >= SPEAKER_ENROLLMENT_SIMILAR_WARNING_THRESHOLD  : 유사 화자(형제/자매 등) -> 경고 로그 + 신규 등록
    그 외                                            : 정상 신규 등록
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
from typing import Callable, List, Optional, Union

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
    # 샘플 안내 문장: 평소 말투로 읽어도 무음 트리밍 후 순수 발화 1.5초 이상이 안정적으로 확보되는 길이.
    # 샘플 수(SPEAKER_ENROLLMENT_REQUIRED_SAMPLES)가 문장 수보다 많으면 순환해서 사용한다.
    SAMPLE_SENTENCES = (
        "오늘 날씨가 정말 화창하고 좋네요",
        "주말에는 가족들과 함께 맛있는 저녁을 먹고 싶어요",
    )

    PROMPT_ASK_SAMPLE = "{name} 님 반가워요! 목소리를 잘 익힐 수 있게 평소 말투로 '{sentence}'라고 말씀해 주세요."
    PROMPT_ASK_NEXT_SAMPLE = "좋아요! 한 번 더, '{sentence}'라고 말씀해 주세요."
    PROMPT_RETRY_SAMPLE = "목소리가 조금 짧았어요. '{sentence}'라고 끝까지 말씀해 주세요."
    PROMPT_COMPLETED = "등록이 완료되었어요! {name} 님, 이제부터 목소리로 바로 알아볼게요."
    PROMPT_COMPLETED_SIMILAR = (
        "등록이 완료되었어요! {name} 님, 다만 {similar_name} 님과 목소리가 많이 비슷해서 가끔 헷갈릴 수 있어요."
    )
    PROMPT_UPDATED_EXISTING = "{existing_name} 님 목소리는 이미 등록되어 있어서, 기존 목소리 정보를 새로 갱신했어요."
    PROMPT_FAILED = "앗, 등록 중에 문제가 생겼어요. 잠시 후에 다시 시도해 주세요."
    PROMPT_CANCELLED = "알겠어요, 목소리 등록을 취소했어요."

    def __init__(
        self,
        speaker_service_instance: SpeakerService = speaker_service,
        session_timeout_sec: Optional[float] = None,
        min_sample_sec: Optional[float] = None,
        required_samples: Optional[int] = None,
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
        self.required_samples = max(
            1, required_samples if required_samples is not None else settings.SPEAKER_ENROLLMENT_REQUIRED_SAMPLES
        )
        self._clock = clock

        self.wake_regex = re.compile(IntentService.WAKE_WORDS_PATTERN)
        self.request_patterns = [re.compile(p) for p in self.ENROLLMENT_REQUEST_PATTERNS]
        self.veto_patterns = [re.compile(p) for p in self.ENROLLMENT_VETO_PATTERNS]
        self.cancel_regex = re.compile(self.CANCEL_PATTERN)
        self.name_prefix_regex = re.compile(self.NAME_PREFIX_PATTERN)
        self.name_suffix_patterns = [re.compile(p) for p in self.NAME_SUFFIX_PATTERNS]

        # 상태 전이는 음성 처리 스레드에서 일어나지만, 조회(is_active)는 다른 스레드에서도
        # 가능하므로 Mutex로 상태/대기 이름/수집 임베딩/마지막 활동 시각을 함께 보호한다.
        self._lock = threading.Lock()
        self._state = EnrollmentState.IDLE
        self._pending_name: Optional[str] = None
        # 평균 대상 임베딩 (이름 발화 0~1개 + 샘플 발화). 샘플 수는 따로 세어 이름 발화 포함 여부와 무관하게 진행한다.
        self._collected_embeddings: List[np.ndarray] = []
        self._collected_sample_count = 0
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
        self._collected_embeddings = []
        self._collected_sample_count = 0
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
            self._collected_embeddings = []
            self._collected_sample_count = 0
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
        - WAITING_FOR_NAME: STT 텍스트에서 이름을 추출하고 이름 발화 임베딩을 평균 대상에 담은 뒤
          WAITING_FOR_VOICE_SAMPLE로 전이.
        - WAITING_FOR_VOICE_SAMPLE: 샘플 발화 임베딩을 수집하고, 필요한 수가 모이면 평균 임베딩으로
          중복 검사 후 DB 적재하고 IDLE 복귀.
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
            return self._handle_name(audio, text, sample_rate)
        return self._handle_voice_sample(audio, pending_name, sample_rate)

    def _sample_sentence(self, index: int) -> str:
        return self.SAMPLE_SENTENCES[index % len(self.SAMPLE_SENTENCES)]

    def _embed_name_utterance(self, audio: Union[np.ndarray, bytes], sample_rate: int) -> Optional[np.ndarray]:
        """
        이름 응답 발화("난 유미야")의 임베딩을 평균 대상으로 추출한다. 이름 발화는 보조 표본이므로
        너무 짧거나(거의 무음) 추출에 실패하면 평균에서만 빼고 온보딩은 그대로 진행한다.
        """
        try:
            wav = self.speaker_service.preprocess(audio, sample_rate=sample_rate)
            duration_sec = self.speaker_service.speech_duration_sec(wav)
            if duration_sec < settings.SPEAKER_ENROLLMENT_MIN_NAME_SAMPLE_SEC:
                logger.info(
                    f"[VoiceEnrollment] 이름 발화가 너무 짧아 평균 임베딩에서 제외 ({duration_sec:.2f}s)"
                )
                return None
            return self.speaker_service.embed_preprocessed(wav)
        except Exception as e:
            logger.warning(f"[VoiceEnrollment] 이름 발화 임베딩 추출 실패, 평균에서 제외: {e}")
            return None

    def _handle_name(
        self, audio: Union[np.ndarray, bytes], text: Optional[str], sample_rate: int
    ) -> EnrollmentReply:
        name = self.extract_name(text or "")
        name_embedding = self._embed_name_utterance(audio, sample_rate) if name is not None else None
        with self._lock:
            if name is None:
                # 재질문도 유효한 응답 대기이므로 타임아웃 기준 시각을 갱신한다
                self._transition_locked(EnrollmentState.WAITING_FOR_NAME)
            else:
                self._pending_name = name
                self._collected_embeddings = [name_embedding] if name_embedding is not None else []
                self._collected_sample_count = 0
                self._transition_locked(EnrollmentState.WAITING_FOR_VOICE_SAMPLE)

        if name is None:
            logger.info(f"[VoiceEnrollment] 이름 추출 실패 (발화: '{text}') -> 재질문")
            return EnrollmentReply(
                speech=self.PROMPT_RETRY_NAME, emotion=EmotionType.CURIOUS, state=EnrollmentState.WAITING_FOR_NAME
            )

        logger.info(f"[VoiceEnrollment] 이름 확인: '{name}' -> WAITING_FOR_VOICE_SAMPLE")
        return EnrollmentReply(
            speech=self.PROMPT_ASK_SAMPLE.format(name=name, sentence=self._sample_sentence(0)),
            emotion=EmotionType.HAPPY,
            state=EnrollmentState.WAITING_FOR_VOICE_SAMPLE,
        )

    def _handle_voice_sample(
        self, audio: Union[np.ndarray, bytes], name: str, sample_rate: int
    ) -> EnrollmentReply:
        # [실제 발화 길이 기준] 캡처 오디오에는 VAD 사전 버퍼/종료 무음(약 1.1초)이 포함되므로, 무음
        # 트리밍 후 길이로 검사해야 실제 말한 부분이 짧은 부실 샘플이 기준 벡터로 등록되지 않는다.
        # 트리밍 결과는 그대로 임베딩에 재사용하고, 길이 미달이면 모델 추론 없이 재요청한다.
        try:
            wav = self.speaker_service.preprocess(audio, sample_rate=sample_rate)
            duration_sec = self.speaker_service.speech_duration_sec(wav)
        except Exception as e:
            logger.warning(f"[VoiceEnrollment] 목소리 샘플 전처리 중 예외 발생: {e}")
            wav, duration_sec = None, None

        with self._lock:
            sample_index = self._collected_sample_count

        if duration_sec is not None and duration_sec < self.min_sample_sec:
            logger.info(
                f"[VoiceEnrollment] 실제 발화가 너무 짧음 ({duration_sec:.2f}s < {self.min_sample_sec:.2f}s) -> 재요청"
            )
            with self._lock:
                self._transition_locked(EnrollmentState.WAITING_FOR_VOICE_SAMPLE)
            return EnrollmentReply(
                speech=self.PROMPT_RETRY_SAMPLE.format(sentence=self._sample_sentence(sample_index)),
                emotion=EmotionType.CURIOUS,
                state=EnrollmentState.WAITING_FOR_VOICE_SAMPLE,
            )

        try:
            if wav is None:
                raise RuntimeError("목소리 샘플 전처리 실패")
            sample_embedding = self.speaker_service.embed_preprocessed(wav)
        except Exception as e:
            logger.warning(f"[VoiceEnrollment] 화자 임베딩 추출 중 예외 발생: {e}")
            return self._fail(name)

        with self._lock:
            self._collected_embeddings.append(sample_embedding)
            self._collected_sample_count += 1
            collected_samples = self._collected_sample_count
            embeddings = list(self._collected_embeddings)
            if collected_samples < self.required_samples:
                self._transition_locked(EnrollmentState.WAITING_FOR_VOICE_SAMPLE)

        if collected_samples < self.required_samples:
            logger.info(
                f"[VoiceEnrollment] 목소리 샘플 {collected_samples}/{self.required_samples} 수집 "
                f"({duration_sec:.2f}s) -> 다음 샘플 요청"
            )
            return EnrollmentReply(
                speech=self.PROMPT_ASK_NEXT_SAMPLE.format(sentence=self._sample_sentence(collected_samples)),
                emotion=EmotionType.HAPPY,
                state=EnrollmentState.WAITING_FOR_VOICE_SAMPLE,
            )

        return self._finalize(name, embeddings)

    def _fail(self, name: Optional[str]) -> EnrollmentReply:
        with self._lock:
            self._reset_locked()
        logger.warning(f"[VoiceEnrollment] 화자 등록 실패 (name: {name}) -> IDLE")
        return EnrollmentReply(speech=self.PROMPT_FAILED, emotion=EmotionType.SAD, state=EnrollmentState.IDLE)

    def _find_closest_profile(self, embedding: np.ndarray):
        """중복 검사용 최근접 기존 화자 조회. 조회 실패는 등록 자체를 막지 않도록 '기존 화자 없음'으로 취급한다."""
        try:
            return self.speaker_service.find_closest_profile(embedding)
        except Exception as e:
            logger.warning(f"[VoiceEnrollment] 중복 화자 검사 중 예외 발생, 검사 생략: {e}")
            return None

    def _finalize(self, name: str, embeddings: List[np.ndarray]) -> EnrollmentReply:
        """수집한 임베딩을 평균/정규화해 중복 검사 후 등록(또는 기존 프로필 갱신)한다."""
        try:
            embedding = SpeakerService.average_embeddings(embeddings)
        except Exception as e:
            logger.warning(f"[VoiceEnrollment] 평균 임베딩 계산 실패: {e}")
            return self._fail(name)

        # [중복 등록 방지] 캐시 무효화 이전(=신규 벡터가 아직 반영되지 않은) 기존 활성 화자와 비교한다.
        closest = self._find_closest_profile(embedding)
        similarity = closest[1] if closest is not None else None

        user_id, display_name = self._generate_user_id(), name
        updated_existing = False
        similar_profile = None
        if closest is not None and similarity >= settings.SPEAKER_ENROLLMENT_DUPLICATE_THRESHOLD:
            # 확실한 중복: 새 user_id를 발급하면 같은 사람의 기억/프로필이 둘로 쪼개지므로, 기존 user_id의
            # 목소리만 새 평균 벡터로 갱신한다. 호칭은 오탐 시 다른 가족 이름을 덮어쓰지 않도록 기존 값을 유지한다.
            existing = closest[0]
            user_id, display_name, updated_existing = existing.user_id, existing.display_name, True
            logger.info(
                f"[VoiceEnrollment] 기존 화자 재등록으로 판단 (유사도 {similarity:.3f} >= "
                f"{settings.SPEAKER_ENROLLMENT_DUPLICATE_THRESHOLD:.2f}, 기존: {existing.user_id}/{existing.display_name}, "
                f"응답 이름: {name}) -> 기존 프로필 목소리 갱신"
            )
        elif closest is not None and similarity >= settings.SPEAKER_ENROLLMENT_SIMILAR_WARNING_THRESHOLD:
            similar_profile = closest[0]
            logger.warning(
                f"[VoiceEnrollment] 유사 화자 경계 구간 (유사도 {similarity:.3f}, 기존: "
                f"{similar_profile.user_id}/{similar_profile.display_name}) -> 신규 등록은 허용하되 식별 마진 주의"
            )

        embedding_list = embedding.tolist()
        try:
            success = upsert_speaker_profile(user_id=user_id, display_name=display_name, embedding=embedding_list)
        except Exception as e:
            logger.warning(f"[VoiceEnrollment] 화자 프로필 적재 중 예외 발생: {e}")
            success = False

        if not success:
            return self._fail(name)

        with self._lock:
            self._reset_locked()

        # [Hot Reload] 캐시를 즉시 무효화해 서버 재기동 없이 다음 턴부터 새 벡터로 식별되도록 하고,
        # 방금 목소리를 들려준 화자를 세션 소프트패스 기준으로 지정한다.
        self.speaker_service.reload_speaker_profiles()
        self.speaker_service.anchor_session_speaker(user_id, display_name)
        similarity_note = f"{similarity:.3f}" if similarity is not None else "기존 화자 없음"
        logger.info(
            f"[VoiceEnrollment] 화자 {'갱신' if updated_existing else '등록'} 완료 (user: {user_id}, name: {display_name}, "
            f"평균 임베딩 {len(embeddings)}개, 최근접 기존 화자 유사도: {similarity_note})"
        )

        if updated_existing:
            speech = self.PROMPT_UPDATED_EXISTING.format(existing_name=display_name)
        elif similar_profile is not None:
            speech = self.PROMPT_COMPLETED_SIMILAR.format(name=name, similar_name=similar_profile.display_name)
        else:
            speech = self.PROMPT_COMPLETED.format(name=name)

        return EnrollmentReply(
            speech=speech,
            emotion=EmotionType.HAPPY,
            state=EnrollmentState.IDLE,
            completed=True,
            user_id=user_id,
            display_name=display_name,
            updated_existing=updated_existing,
            similar_to_user_id=similar_profile.user_id if similar_profile is not None else None,
        )

    @staticmethod
    def _generate_user_id() -> str:
        return f"user_{datetime.now().strftime('%Y%m%d%H%M%S')}"


# Spring Bean 스타일 전역 싱글톤 등록
voice_enrollment_service = VoiceEnrollmentService()
