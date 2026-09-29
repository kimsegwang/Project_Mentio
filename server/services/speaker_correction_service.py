"""
server/services/speaker_correction_service.py
대화형 호칭 정정(Speaker Disambiguation) 서비스.

형제/자매처럼 목소리가 비슷한 가족을 로봇이 잘못 알아봤을 때, 사용자가 "나 민수인데?",
"나 지훈이야", "나 민수 아니고 지훈이야"처럼 말로 바로잡으면 등록된 활성 화자 중 해당 이름을
찾아 세션 화자를 즉시 전환하고 사과 멘트를 만든다.

- LLM 호출 없는 정규식 룰로만 판별해 지연을 추가하지 않는다 (IntentService/VoiceEnrollment와 동일 원칙).
- 발화 전체가 정정 표현일 때만 인정한다. "나 민수인데 오늘 축구했어"처럼 내용이 이어지는 발화는
  일반 대화로 흘려보내 사용자 발화 내용이 사과 멘트로 묻히지 않도록 한다.
- 추출한 이름이 등록된 활성 화자와 정확히 1명 일치하고, 그 화자가 현재 식별된 화자와 다를 때만
  정정으로 본다. 이미 맞게 식별된 경우("나 민수야" -> 민수)나 등록되지 않은 이름("나 학생인데")은
  일반 대화 파이프라인으로 넘긴다.
- 오디오/TTS I/O는 다루지 않는 순수 도메인 서비스다.
"""
import logging
import re
from typing import List, Optional

from server.schemas.speaker import SpeakerCorrection
from server.services.intent_service import IntentService
from server.services.speaker_service import SpeakerService, speaker_service

logger = logging.getLogger(__name__)

_SELF_PRONOUN = r"(?:나|난|나는|저|전|저는|내가|제가)"
_COPULA_SUFFIX = r"(?:이야|야|인데|이잖아|잖아|이라니까|라니까|이에요|예요|에요|입니다|이거든|거든)(?:요)?"


class SpeakerCorrectionService:
    # 정정 패턴 (정규화 후 발화 전체 매칭). name 그룹이 "실제 화자" 이름이다.
    CORRECTION_PATTERNS = [
        # "나 민수 아니고 지훈이야", "민수가 아니라 지훈이라니까"
        rf"^(?:{_SELF_PRONOUN}\s*)?(?P<wrong>\S+?)(?:이|가)?\s*아니(?:고|라)\s*(?P<name>\S+?){_COPULA_SUFFIX}$",
        # "나 민수인데", "나 지훈이야", "저 엄마예요"
        rf"^{_SELF_PRONOUN}\s+(?P<name>\S+?){_COPULA_SUFFIX}$",
    ]

    PROMPT_CORRECTED = "앗, {vocative} 미안해! 목소리가 비슷해서 착각했어."

    def __init__(self, speaker_service_instance: SpeakerService = speaker_service):
        self.speaker_service = speaker_service_instance
        self.wake_regex = re.compile(IntentService.WAKE_WORDS_PATTERN)
        self.correction_patterns = [re.compile(p) for p in self.CORRECTION_PATTERNS]

    def _normalize(self, text: str) -> str:
        clean_text = re.sub(r"[^\w\s]", " ", (text or "").strip())
        clean_text = re.sub(r"\s+", " ", clean_text).strip()
        target_text = self.wake_regex.sub("", clean_text).strip()
        return target_text or clean_text

    def extract_claimed_name(self, text: str) -> Optional[str]:
        """정정 발화에서 "실제 화자" 이름을 추출한다. 정정 표현이 아니면 None."""
        target_text = self._normalize(text)
        for pattern in self.correction_patterns:
            match = pattern.match(target_text)
            if match:
                return match.group("name")
        return None

    @staticmethod
    def _name_candidates(name: str) -> List[str]:
        """"지훈이야" -> "지훈이"처럼 애칭 접미 '이'가 붙은 채 추출될 수 있어 떼어낸 후보도 함께 시도한다."""
        candidates = [name]
        if len(name) > 1 and name.endswith("이"):
            candidates.append(name[:-1])
        return candidates

    def detect(self, text: str, current_user_id: Optional[str]) -> Optional[SpeakerCorrection]:
        """
        정정 발화이면서 이름이 현재 식별 화자와 다른 활성 화자 1명과 일치하면 SpeakerCorrection을 반환한다.
        화자 프로필 조회 실패 등 예외는 정정 없음(None)으로 처리해 일반 대화를 막지 않는다.
        """
        spoken_name = self.extract_claimed_name(text)
        if spoken_name is None:
            return None

        try:
            profile = None
            for candidate in self._name_candidates(spoken_name):
                profile = self.speaker_service.find_active_profile_by_name(candidate)
                if profile is not None:
                    break
        except Exception as e:
            logger.warning(f"[SpeakerCorrection] 화자 이름 매칭 중 예외 발생, 정정 생략: {e}")
            return None

        if profile is None:
            logger.info(f"[SpeakerCorrection] 정정 표현이지만 등록된 화자 이름과 불일치 ('{spoken_name}') -> 일반 대화")
            return None
        if profile.user_id == current_user_id:
            return None

        return SpeakerCorrection(
            user_id=profile.user_id,
            display_name=profile.display_name,
            spoken_name=spoken_name,
            previous_user_id=current_user_id,
        )

    def apply(self, correction: SpeakerCorrection) -> str:
        """세션 화자를 정정 대상으로 강제 전환하고, TTS로 출력할 사과 멘트를 반환한다."""
        self.speaker_service.anchor_session_speaker(correction.user_id, correction.display_name)
        logger.info(
            f"[SpeakerCorrection] 호칭 정정 -> 세션 화자 강제 전환 "
            f"({correction.previous_user_id} -> {correction.user_id}/{correction.display_name})"
        )
        return self.PROMPT_CORRECTED.format(vocative=self.vocative(correction.display_name))

    @staticmethod
    def vocative(name: str) -> str:
        """호격 조사: 받침 있으면 '아'(지훈아), 없으면 '야'(민수야). 한글 음절이 아니면 쉼표만 붙인다."""
        if not name:
            return name
        last = name[-1]
        if not ("가" <= last <= "힣"):
            return f"{name},"
        has_final_consonant = (ord(last) - ord("가")) % 28 != 0
        return f"{name}{'아' if has_final_consonant else '야'}"


# Spring Bean 스타일 전역 싱글톤 등록
speaker_correction_service = SpeakerCorrectionService()
