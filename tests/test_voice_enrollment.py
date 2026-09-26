"""
tests/test_voice_enrollment.py
대화형 음성 온보딩(VoiceEnrollmentService + AIWorker 연동) 검증 pytest 단위 테스트.

검증 대상:
- 온보딩 상태 전이: IDLE -> WAITING_FOR_NAME -> WAITING_FOR_VOICE_SAMPLE -> IDLE(등록 완료)
- 등록 요청/취소/이름 추출 정규식 룰
- 세션 타임아웃 만료(바이패스 창이 무기한 열려 있지 않음)
- 화자 식별 차단(Fail-Close) 바이패스: 온보딩 중에만 identify_speaker를 건너뜀
- 프로필 생성(upsert_speaker_profile) 및 캐시 무효화(Hot Reload)로 다음 턴부터 신규 화자 식별
Resemblyzer 실제 모델/DB는 사용하지 않고 가짜 임베딩과 인메모리 프로필 목록으로 대체한다.
"""
import re
from unittest.mock import Mock

import numpy as np
import pytest

import server.services.speaker_service as speaker_service_module
import server.services.voice_enrollment_service as enrollment_module
import server.workers.ai_worker as ai_worker_module
from server.repositories.speaker_repository import SpeakerEmbeddingRecord
from server.schemas.action import EmotionType, LLMResponse, TriggerType
from server.schemas.speaker import EnrollmentState, SpeakerIdentificationResult
from server.services.speaker_service import SpeakerService
from server.services.voice_enrollment_service import VoiceEnrollmentService
from server.workers.ai_worker import AIWorker

SAMPLE_RATE = 16000
LONG_AUDIO = np.zeros(SAMPLE_RATE * 3, dtype=np.float32)  # 3.0초
SHORT_AUDIO = np.zeros(SAMPLE_RATE // 2, dtype=np.float32)  # 0.5초
NEW_SPEAKER_EMBEDDING = np.array([0.0, 1.0], dtype=np.float32)
EXISTING_SPEAKER_EMBEDDING = np.array([1.0, 0.0], dtype=np.float32)


class FakeClock:
    def __init__(self, t: float = 1000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


@pytest.fixture()
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture()
def fake_speaker_service() -> Mock:
    svc = Mock()
    svc.audio_duration_sec.side_effect = SpeakerService.audio_duration_sec
    svc.embed.return_value = NEW_SPEAKER_EMBEDDING
    return svc


@pytest.fixture()
def upsert_mock(monkeypatch) -> Mock:
    mock = Mock(return_value=True)
    monkeypatch.setattr(enrollment_module, "upsert_speaker_profile", mock)
    return mock


@pytest.fixture()
def enrollment(fake_speaker_service, clock, upsert_mock) -> VoiceEnrollmentService:
    return VoiceEnrollmentService(
        speaker_service_instance=fake_speaker_service,
        session_timeout_sec=30.0,
        min_sample_sec=1.5,
        clock=clock,
    )


# --- 등록 요청 / 취소 / 이름 추출 룰 ---

@pytest.mark.parametrize(
    "text",
    [
        "목소리 등록할래",
        "내 목소리 기억해줘",
        "새 화자 등록",
        "멘티오야, 내 목소리 좀 등록해 줘!",
        "새로운 가족 추가해줘",
        "화자 등록해줘",
    ],
)
def test_detects_enrollment_request(enrollment, text):
    assert enrollment.is_enrollment_request(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "오늘 날씨 어때?",
        "내 목소리 기억나?",
        "목소리 등록하지 마",
        "지금 몇 시야",
        "",
    ],
)
def test_ignores_non_enrollment_utterances(enrollment, text):
    assert enrollment.is_enrollment_request(text) is False


@pytest.mark.parametrize(
    "text, expected",
    [
        ("민수", "민수"),
        ("저는 민수예요", "민수"),
        ("제 이름은 김철수입니다.", "김철수"),
        ("지현이라고 해요", "지현"),
        ("첫째라고 불러줘", "첫째"),
        ("난 엄마야", "엄마"),
        ("멘티오야 나는 아빠야", "아빠"),
        ("민수 님이요", "민수"),
    ],
)
def test_extract_name_strips_prefix_and_suffix(enrollment, text, expected):
    assert enrollment.extract_name(text) == expected


@pytest.mark.parametrize("text", ["", "   ", "오늘은 정말 날씨가 좋아서 산책을 다녀왔어요"])
def test_extract_name_returns_none_for_empty_or_sentence(enrollment, text):
    assert enrollment.extract_name(text) is None


@pytest.mark.parametrize("text", ["취소", "등록 취소", "그만해", "안 할래", "됐어요"])
def test_detects_cancel_request(enrollment, text):
    assert enrollment.is_cancel_request(text) is True


def test_cancel_requires_whole_utterance_so_sample_sentence_is_not_cancel(enrollment):
    assert enrollment.is_cancel_request("이제 그만 자러 가야겠다") is False


# --- 상태 전이 ---

def test_initial_state_is_idle(enrollment):
    assert enrollment.state == EnrollmentState.IDLE
    assert enrollment.is_active() is False


def test_handle_turn_returns_none_when_idle(enrollment):
    assert enrollment.handle_turn(LONG_AUDIO, "민수") is None


def test_start_transitions_to_waiting_for_name(enrollment):
    reply = enrollment.start()

    assert reply.state == EnrollmentState.WAITING_FOR_NAME
    assert reply.speech == "반가워요! 등록하실 분의 성함이나 호칭을 말씀해 주세요."
    assert enrollment.is_active() is True


def test_name_turn_transitions_to_waiting_for_voice_sample(enrollment):
    enrollment.start()

    reply = enrollment.handle_turn(LONG_AUDIO, "저는 민수예요")

    assert reply.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    assert reply.speech == "민수 님 반가워요! '오늘 날씨가 참 좋다'처럼 평소 말투로 한 문장 말씀해 주세요."
    assert enrollment.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE


def test_unrecognized_name_re_asks_and_stays_waiting_for_name(enrollment):
    enrollment.start()

    reply = enrollment.handle_turn(LONG_AUDIO, "")

    assert reply.state == EnrollmentState.WAITING_FOR_NAME
    assert "다시" in reply.speech
    assert enrollment.state == EnrollmentState.WAITING_FOR_NAME


def test_short_voice_sample_re_requests_without_registering(enrollment, upsert_mock, fake_speaker_service):
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")

    reply = enrollment.handle_turn(SHORT_AUDIO, "좋다")

    assert reply.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    assert reply.completed is False
    upsert_mock.assert_not_called()
    fake_speaker_service.embed.assert_not_called()


def test_voice_sample_registers_profile_and_invalidates_cache(enrollment, upsert_mock, fake_speaker_service):
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")

    reply = enrollment.handle_turn(LONG_AUDIO, "오늘 날씨가 참 좋다")

    assert reply.completed is True
    assert reply.state == EnrollmentState.IDLE
    assert reply.display_name == "민수"
    assert re.fullmatch(r"user_\d{14}", reply.user_id)
    assert reply.speech == "등록이 완료되었어요! 민수 님, 이제부터 목소리로 바로 알아볼게요."

    fake_speaker_service.embed.assert_called_once()
    upsert_mock.assert_called_once_with(
        user_id=reply.user_id, display_name="민수", embedding=NEW_SPEAKER_EMBEDDING.tolist()
    )
    fake_speaker_service.reload_speaker_profiles.assert_called_once()
    fake_speaker_service.anchor_session_speaker.assert_called_once_with(reply.user_id, "민수")
    assert enrollment.is_active() is False


def test_db_failure_resets_to_idle_without_cache_invalidation(enrollment, upsert_mock, fake_speaker_service):
    upsert_mock.return_value = False
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")

    reply = enrollment.handle_turn(LONG_AUDIO, "오늘 날씨가 참 좋다")

    assert reply.completed is False
    assert reply.emotion == EmotionType.SAD
    assert enrollment.state == EnrollmentState.IDLE
    fake_speaker_service.reload_speaker_profiles.assert_not_called()


def test_embedding_exception_is_handled_as_failure(enrollment, upsert_mock, fake_speaker_service):
    fake_speaker_service.embed.side_effect = RuntimeError("resemblyzer backend error")
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")

    reply = enrollment.handle_turn(LONG_AUDIO, "오늘 날씨가 참 좋다")

    assert reply.completed is False
    assert enrollment.state == EnrollmentState.IDLE
    upsert_mock.assert_not_called()


@pytest.mark.parametrize("advance_to_sample_stage", [False, True])
def test_cancel_returns_to_idle_from_any_stage(enrollment, upsert_mock, advance_to_sample_stage):
    enrollment.start()
    if advance_to_sample_stage:
        enrollment.handle_turn(LONG_AUDIO, "민수")

    reply = enrollment.handle_turn(LONG_AUDIO, "취소")

    assert reply.state == EnrollmentState.IDLE
    assert "취소" in reply.speech
    assert enrollment.is_active() is False
    upsert_mock.assert_not_called()


def test_restart_request_during_session_resets_pending_name(enrollment):
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")

    enrollment.start()

    assert enrollment.state == EnrollmentState.WAITING_FOR_NAME


# --- 세션 타임아웃 (바이패스 창 제한) ---

def test_session_expires_after_timeout(enrollment, clock):
    enrollment.start()
    clock.t += 31.0

    assert enrollment.is_active() is False
    assert enrollment.handle_turn(LONG_AUDIO, "민수") is None


def test_each_turn_refreshes_session_timeout(enrollment, clock):
    enrollment.start()
    clock.t += 20.0
    enrollment.handle_turn(LONG_AUDIO, "민수")  # 이름 응답으로 타임아웃 기준 시각 갱신
    clock.t += 20.0  # 시작 기준으로는 40초 경과지만, 마지막 활동 기준으로는 20초

    assert enrollment.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE


# --- AIWorker 연동: 차단 바이패스 / 파이프라인 분기 ---

@pytest.fixture()
def worker_parts(monkeypatch, enrollment, fake_speaker_service):
    mock_brain = Mock()
    mock_brain.infer_action.return_value = LLMResponse(emotion=EmotionType.HAPPY, speech="일반 응답")
    mock_tts = Mock()
    mock_tts.synthesize.return_value = b""

    fake_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=True, user_id="dad", display_name="아빠", similarity=0.9, threshold=0.65
    )

    w = AIWorker(
        brain_service_instance=mock_brain,
        tts_service_instance=mock_tts,
        audio_player_instance=Mock(),
        speaker_service_instance=fake_speaker_service,
        enrollment_service_instance=enrollment,
    )

    log_mock = Mock()
    submit_mock = Mock()
    intent_mock = Mock(return_value=(TriggerType.VOICE_CHAT.value, False))
    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", log_mock)
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", submit_mock)
    monkeypatch.setattr(ai_worker_module.intent_service, "analyze_voice_intent", intent_mock)
    monkeypatch.setattr(w, "_retrieve_memory_context", lambda user_text, user_id: "")
    return w, log_mock, submit_mock, intent_mock


def _set_stt(monkeypatch, text):
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: text)


def test_worker_starts_enrollment_on_request_and_skips_llm_and_memory(worker_parts, monkeypatch, enrollment):
    worker, log_mock, submit_mock, intent_mock = worker_parts
    _set_stt(monkeypatch, "내 목소리 기억해줘")

    action = worker.process_voice_interaction(LONG_AUDIO)

    assert action.speech == VoiceEnrollmentService.PROMPT_ASK_NAME
    assert enrollment.state == EnrollmentState.WAITING_FOR_NAME
    worker.brain_service.infer_action.assert_not_called()
    intent_mock.assert_not_called()
    submit_mock.assert_not_called()
    assert log_mock.call_args.kwargs["trigger_type"] == TriggerType.VOICE_ENROLLMENT.value
    queued_action, trigger, _ = worker.poll_result()
    assert trigger == TriggerType.VOICE_ENROLLMENT.value
    assert queued_action.speech == VoiceEnrollmentService.PROMPT_ASK_NAME


def test_worker_bypasses_speaker_gate_during_enrollment(worker_parts, monkeypatch, enrollment, fake_speaker_service):
    worker, _, _, _ = worker_parts
    enrollment.start()
    # 신규 화자는 아직 미등록이라 식별을 돌리면 차단되는 상황
    fake_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=False, similarity=0.3, threshold=0.65
    )
    _set_stt(monkeypatch, "저는 민수예요")

    action = worker.process_voice_interaction(LONG_AUDIO)

    assert action is not None
    assert action.speech.startswith("민수 님 반가워요!")
    fake_speaker_service.identify_speaker.assert_not_called()


def test_worker_still_blocks_unidentified_speaker_when_not_enrolling(
    worker_parts, monkeypatch, enrollment, fake_speaker_service
):
    """온보딩 세션 밖에서는 미등록 화자가 등록 요청 문구를 말해도 STT 이전에 차단된다."""
    worker, _, _, _ = worker_parts
    fake_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=False, similarity=0.3, threshold=0.65
    )
    stt_mock = Mock(return_value="내 목소리 기억해줘")
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", stt_mock)

    result = worker.process_voice_interaction(LONG_AUDIO)

    assert result is None
    stt_mock.assert_not_called()
    assert enrollment.is_active() is False


def test_worker_restores_speaker_gate_after_session_timeout(
    worker_parts, monkeypatch, enrollment, fake_speaker_service, clock
):
    worker, _, _, _ = worker_parts
    enrollment.start()
    clock.t += 31.0
    fake_speaker_service.identify_speaker.return_value = SpeakerIdentificationResult(
        is_match=False, similarity=0.3, threshold=0.65
    )
    _set_stt(monkeypatch, "민수")

    result = worker.process_voice_interaction(LONG_AUDIO)

    assert result is None
    fake_speaker_service.identify_speaker.assert_called_once()


def test_worker_does_not_route_normal_chat_into_enrollment(worker_parts, monkeypatch, enrollment):
    worker, _, submit_mock, _ = worker_parts
    _set_stt(monkeypatch, "오늘 축구했어")

    action = worker.process_voice_interaction(LONG_AUDIO)

    assert action.speech == "일반 응답"
    assert enrollment.is_active() is False
    submit_mock.assert_called_once_with("오늘 축구했어", user_id="dad")


# --- 통합: 실제 SpeakerService 캐시 무효화로 다음 턴부터 신규 화자 식별 (Hot Reload) ---

def test_full_voice_onboarding_hot_reloads_new_speaker_for_next_turn(monkeypatch, clock):
    fake_db = [
        SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=EXISTING_SPEAKER_EMBEDDING.tolist())
    ]

    def fake_upsert(user_id, display_name, embedding):
        fake_db.append(SpeakerEmbeddingRecord(user_id=user_id, display_name=display_name, embedding=embedding))
        return True

    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", True)
    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", lambda: list(fake_db))
    monkeypatch.setattr(enrollment_module, "upsert_speaker_profile", fake_upsert)

    real_speaker_service = SpeakerService(similarity_threshold=0.65)
    current_voice = {"embedding": EXISTING_SPEAKER_EMBEDDING}
    monkeypatch.setattr(
        real_speaker_service, "embed", lambda audio, sample_rate=16000: current_voice["embedding"].copy()
    )
    enrollment = VoiceEnrollmentService(speaker_service_instance=real_speaker_service, clock=clock)

    mock_tts = Mock()
    mock_tts.synthesize.return_value = b""
    mock_brain = Mock()
    mock_brain.infer_action.return_value = LLMResponse(emotion=EmotionType.HAPPY, speech="일반 응답")
    worker = AIWorker(
        brain_service_instance=mock_brain,
        tts_service_instance=mock_tts,
        audio_player_instance=Mock(),
        speaker_service_instance=real_speaker_service,
        enrollment_service_instance=enrollment,
    )
    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", Mock())
    submit_mock = Mock()
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", submit_mock)
    monkeypatch.setattr(
        ai_worker_module.intent_service, "analyze_voice_intent", lambda text: (TriggerType.VOICE_CHAT.value, False)
    )
    monkeypatch.setattr(worker, "_retrieve_memory_context", lambda user_text, user_id: "")

    # 1) 등록된 아빠가 등록 요청 -> 캐시에 아빠 프로필 1명 로딩
    _set_stt(monkeypatch, "새 화자 등록")
    worker.process_voice_interaction(LONG_AUDIO)
    assert enrollment.state == EnrollmentState.WAITING_FOR_NAME

    # 2) 신규 화자(미등록 목소리)가 이름 응답 -> 바이패스로 통과
    current_voice["embedding"] = NEW_SPEAKER_EMBEDDING
    _set_stt(monkeypatch, "저는 민수예요")
    worker.process_voice_interaction(LONG_AUDIO)
    assert enrollment.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE

    # 3) 목소리 샘플 -> DB 적재 + 캐시 무효화
    _set_stt(monkeypatch, "오늘 날씨가 참 좋다")
    action = worker.process_voice_interaction(LONG_AUDIO)
    assert action.speech == "등록이 완료되었어요! 민수 님, 이제부터 목소리로 바로 알아볼게요."
    assert len(fake_db) == 2
    submit_mock.assert_not_called()  # 온보딩 발화는 장기 기억으로 적재하지 않음

    # 4) 서버 재기동 없이 다음 턴부터 신규 화자로 식별되어 일반 대화 파이프라인 진입
    _set_stt(monkeypatch, "나 오늘 축구했어")
    action = worker.process_voice_interaction(LONG_AUDIO)
    new_user_id = fake_db[-1].user_id
    assert action.speech == "일반 응답"
    assert mock_brain.infer_action.call_args.kwargs["display_name"] == "민수"
    submit_mock.assert_called_once_with("나 오늘 축구했어", user_id=new_user_id)


def test_anchor_session_speaker_sets_soft_pass_identity(monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_VERIFICATION_ENABLED", True)
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC", 10.0)
    monkeypatch.setattr(
        speaker_service_module,
        "get_active_speaker_embeddings",
        lambda: [
            SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=EXISTING_SPEAKER_EMBEDDING.tolist()),
            SpeakerEmbeddingRecord(user_id="user_new", display_name="민수", embedding=NEW_SPEAKER_EMBEDDING.tolist()),
        ],
    )
    service = SpeakerService(similarity_threshold=0.65)
    service.anchor_session_speaker("user_new", "민수")
    # 두 후보 모두와 낮은 유사도의 짧은 맞장구 -> 소프트패스로 방금 등록한 신규 화자에게 귀속
    monkeypatch.setattr(service, "embed", lambda audio, sample_rate=16000: np.array([-1.0, -1.0]))

    result = service.identify_speaker(LONG_AUDIO)

    assert result.soft_passed is True
    assert result.user_id == "user_new"
    assert result.display_name == "민수"
