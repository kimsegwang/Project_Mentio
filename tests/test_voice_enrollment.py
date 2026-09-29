"""
tests/test_voice_enrollment.py
대화형 음성 온보딩(VoiceEnrollmentService + AIWorker 연동) 검증 pytest 단위 테스트.

검증 대상:
- 온보딩 상태 전이: IDLE -> WAITING_FOR_NAME -> WAITING_FOR_VOICE_SAMPLE -> IDLE(등록 완료)
- 등록 요청/취소/이름 추출 정규식 룰
- 세션 타임아웃 만료(바이패스 창이 무기한 열려 있지 않음)
- 화자 식별 차단(Fail-Close) 바이패스: 온보딩 중에만 identify_speaker를 건너뜀
- 프로필 생성(upsert_speaker_profile) 및 캐시 무효화(Hot Reload)로 다음 턴부터 신규 화자 식별
- 다중 발화 평균화: 이름 발화 + 샘플 2개 임베딩 평균 후 L2 정규화
- 중복/유사 화자 등록 방지: 0.85 이상 기존 프로필 갱신 / 0.75~0.85 경고 후 신규 등록 / 미만 정상 등록
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
NEAR_SILENT_AUDIO = np.zeros(SAMPLE_RATE // 5, dtype=np.float32)  # 0.2초 (이름 발화 평균 제외 기준 미만)
SAMPLE_1 = "오늘 날씨가 정말 화창하고 좋네요"
SAMPLE_2 = "주말에는 가족들과 함께 맛있는 저녁을 먹고 싶어요"
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
    # 전처리(무음 트리밍)는 항등 함수로 두어, 테스트 오디오 길이 = 트리밍 후 실제 발화 길이로 간주한다
    svc.preprocess.side_effect = lambda audio, sample_rate=16000: audio
    svc.speech_duration_sec.side_effect = SpeakerService.speech_duration_sec
    svc.embed_preprocessed.return_value = NEW_SPEAKER_EMBEDDING
    svc.find_closest_profile.return_value = None  # 기본: 기존 활성 화자 없음
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
    assert reply.speech == f"민수 님 반가워요! 목소리를 잘 익힐 수 있게 평소 말투로 '{SAMPLE_1}'라고 말씀해 주세요."
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
    fake_speaker_service.embed_preprocessed.reset_mock()  # 이름 발화 임베딩 호출은 제외

    reply = enrollment.handle_turn(SHORT_AUDIO, "좋다")

    assert reply.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    assert reply.completed is False
    assert SAMPLE_1 in reply.speech
    upsert_mock.assert_not_called()
    fake_speaker_service.embed_preprocessed.assert_not_called()


def test_first_sample_asks_for_second_sample_without_registering(enrollment, upsert_mock, fake_speaker_service):
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")

    reply = enrollment.handle_turn(LONG_AUDIO, SAMPLE_1)

    assert reply.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    assert reply.completed is False
    assert reply.speech == f"좋아요! 한 번 더, '{SAMPLE_2}'라고 말씀해 주세요."
    upsert_mock.assert_not_called()
    fake_speaker_service.find_closest_profile.assert_not_called()


def test_short_second_sample_re_requests_second_sentence(enrollment, upsert_mock):
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")
    enrollment.handle_turn(LONG_AUDIO, SAMPLE_1)

    reply = enrollment.handle_turn(SHORT_AUDIO, "주말에")

    assert reply.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    assert SAMPLE_2 in reply.speech
    upsert_mock.assert_not_called()
    # 재시도 후 두 번째 샘플이 들어오면 정상 완료 (짧은 샘플은 개수에 포함되지 않음)
    assert enrollment.handle_turn(LONG_AUDIO, SAMPLE_2).completed is True


def test_voice_sample_registers_profile_and_invalidates_cache(enrollment, upsert_mock, fake_speaker_service):
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")
    enrollment.handle_turn(LONG_AUDIO, SAMPLE_1)

    reply = enrollment.handle_turn(LONG_AUDIO, SAMPLE_2)

    assert reply.completed is True
    assert reply.state == EnrollmentState.IDLE
    assert reply.display_name == "민수"
    assert reply.updated_existing is False
    assert reply.similar_to_user_id is None
    assert re.fullmatch(r"user_\d{14}", reply.user_id)
    assert reply.speech == "등록이 완료되었어요! 민수 님, 이제부터 목소리로 바로 알아볼게요."

    assert fake_speaker_service.embed_preprocessed.call_count == 3  # 이름 + 샘플 2개
    upsert_mock.assert_called_once_with(
        user_id=reply.user_id, display_name="민수", embedding=NEW_SPEAKER_EMBEDDING.tolist()
    )
    fake_speaker_service.reload_speaker_profiles.assert_called_once()
    fake_speaker_service.anchor_session_speaker.assert_called_once_with(reply.user_id, "민수")
    assert enrollment.is_active() is False


# --- 다중 발화 평균화 ---

def _complete_onboarding(enrollment, name_audio=LONG_AUDIO):
    enrollment.start()
    enrollment.handle_turn(name_audio, "난 유미야")
    enrollment.handle_turn(LONG_AUDIO, SAMPLE_1)
    return enrollment.handle_turn(LONG_AUDIO, SAMPLE_2)


def test_registers_l2_normalized_average_of_name_and_two_samples(enrollment, upsert_mock, fake_speaker_service):
    name_emb = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    sample1_emb = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    sample2_emb = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    fake_speaker_service.embed_preprocessed.side_effect = [name_emb, sample1_emb, sample2_emb]

    reply = _complete_onboarding(enrollment)

    assert reply.completed is True
    assert reply.display_name == "유미"
    registered = np.array(upsert_mock.call_args.kwargs["embedding"])
    np.testing.assert_allclose(registered, np.ones(3) / np.sqrt(3), rtol=1e-6)
    assert np.linalg.norm(registered) == pytest.approx(1.0)


def test_near_silent_name_utterance_is_excluded_from_average(enrollment, upsert_mock, fake_speaker_service):
    sample1_emb = np.array([1.0, 0.0], dtype=np.float32)
    sample2_emb = np.array([0.0, 1.0], dtype=np.float32)
    fake_speaker_service.embed_preprocessed.side_effect = [sample1_emb, sample2_emb]

    reply = _complete_onboarding(enrollment, name_audio=NEAR_SILENT_AUDIO)

    assert reply.completed is True
    assert fake_speaker_service.embed_preprocessed.call_count == 2  # 이름 발화는 임베딩하지 않음
    registered = np.array(upsert_mock.call_args.kwargs["embedding"])
    np.testing.assert_allclose(registered, np.array([1.0, 1.0]) / np.sqrt(2), rtol=1e-6)


def test_name_embedding_failure_does_not_block_enrollment(enrollment, upsert_mock, fake_speaker_service):
    fake_speaker_service.embed_preprocessed.side_effect = [
        RuntimeError("name embed error"),
        NEW_SPEAKER_EMBEDDING,
        NEW_SPEAKER_EMBEDDING,
    ]

    reply = _complete_onboarding(enrollment)

    assert reply.completed is True
    upsert_mock.assert_called_once()


def test_average_embeddings_returns_unit_norm_vector():
    result = SpeakerService.average_embeddings([[3.0, 0.0], np.array([0.0, 4.0]), [1.0, 1.0]])

    assert result.dtype == np.float32
    assert np.linalg.norm(result) == pytest.approx(1.0)
    np.testing.assert_allclose(result, np.array([4.0, 5.0]) / np.linalg.norm([4.0, 5.0]), rtol=1e-6)


@pytest.mark.parametrize("embeddings", [[], [[1.0, 0.0], [-1.0, 0.0]]])
def test_average_embeddings_rejects_empty_or_cancelling_inputs(embeddings):
    with pytest.raises(ValueError):
        SpeakerService.average_embeddings(embeddings)


# --- 중복/유사 화자 등록 방지 ---

DAD = SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=EXISTING_SPEAKER_EMBEDDING.tolist())


@pytest.mark.parametrize("similarity", [0.85, 0.93])
def test_duplicate_speaker_updates_existing_profile_instead_of_new_user(
    enrollment, upsert_mock, fake_speaker_service, similarity
):
    fake_speaker_service.find_closest_profile.return_value = (DAD, similarity)

    reply = _complete_onboarding(enrollment)

    assert reply.completed is True
    assert reply.updated_existing is True
    assert reply.user_id == "dad"
    assert reply.display_name == "아빠"  # 응답한 이름("유미")으로 기존 호칭을 덮어쓰지 않음
    assert reply.speech == "아빠 님 목소리는 이미 등록되어 있어서, 기존 목소리 정보를 새로 갱신했어요."
    upsert_mock.assert_called_once_with(user_id="dad", display_name="아빠", embedding=NEW_SPEAKER_EMBEDDING.tolist())
    fake_speaker_service.reload_speaker_profiles.assert_called_once()
    fake_speaker_service.anchor_session_speaker.assert_called_once_with("dad", "아빠")


@pytest.mark.parametrize("similarity", [0.75, 0.8, 0.849])
def test_similar_speaker_registers_new_user_with_warning(
    enrollment, upsert_mock, fake_speaker_service, similarity, caplog
):
    fake_speaker_service.find_closest_profile.return_value = (DAD, similarity)

    with caplog.at_level("WARNING", logger=enrollment_module.logger.name):
        reply = _complete_onboarding(enrollment)

    assert reply.completed is True
    assert reply.updated_existing is False
    assert reply.similar_to_user_id == "dad"
    assert reply.user_id != "dad"
    assert reply.display_name == "유미"
    assert reply.speech == "등록이 완료되었어요! 유미 님, 다만 아빠 님과 목소리가 많이 비슷해서 가끔 헷갈릴 수 있어요."
    assert upsert_mock.call_args.kwargs["user_id"] == reply.user_id
    assert "유사 화자 경계" in caplog.text


def test_dissimilar_speaker_registers_normally(enrollment, upsert_mock, fake_speaker_service):
    fake_speaker_service.find_closest_profile.return_value = (DAD, 0.749)

    reply = _complete_onboarding(enrollment)

    assert reply.completed is True
    assert reply.updated_existing is False
    assert reply.similar_to_user_id is None
    assert reply.speech == "등록이 완료되었어요! 유미 님, 이제부터 목소리로 바로 알아볼게요."


def test_duplicate_check_failure_falls_back_to_new_registration(enrollment, upsert_mock, fake_speaker_service):
    fake_speaker_service.find_closest_profile.side_effect = RuntimeError("cache error")

    reply = _complete_onboarding(enrollment)

    assert reply.completed is True
    assert reply.updated_existing is False
    upsert_mock.assert_called_once()


def test_find_closest_profile_returns_best_active_profile(monkeypatch):
    mom = SpeakerEmbeddingRecord(user_id="mom", display_name="엄마", embedding=[0.6, 0.8])
    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", lambda: [DAD, mom])
    service = SpeakerService()

    profile, similarity = service.find_closest_profile(np.array([0.0, 1.0]))

    assert profile.user_id == "mom"
    assert similarity == pytest.approx(0.8)


def test_find_closest_profile_returns_none_without_profiles(monkeypatch):
    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", lambda: [])

    assert SpeakerService().find_closest_profile(np.array([0.0, 1.0])) is None


def test_db_failure_resets_to_idle_without_cache_invalidation(enrollment, upsert_mock, fake_speaker_service):
    upsert_mock.return_value = False
    enrollment.start()
    enrollment.handle_turn(LONG_AUDIO, "민수")
    enrollment.handle_turn(LONG_AUDIO, SAMPLE_1)

    reply = enrollment.handle_turn(LONG_AUDIO, SAMPLE_2)

    assert reply.completed is False
    assert reply.emotion == EmotionType.SAD
    assert enrollment.state == EnrollmentState.IDLE
    fake_speaker_service.reload_speaker_profiles.assert_not_called()


def test_embedding_exception_is_handled_as_failure(enrollment, upsert_mock, fake_speaker_service):
    fake_speaker_service.embed_preprocessed.side_effect = RuntimeError("resemblyzer backend error")
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
    submit_mock.assert_called_once_with("오늘 축구했어", user_id="dad", speaker_ambiguous=False)


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
    monkeypatch.setattr(real_speaker_service, "preprocess", lambda audio, sample_rate=16000: audio)
    monkeypatch.setattr(real_speaker_service, "embed_preprocessed", lambda wav: current_voice["embedding"].copy())
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

    # 3) 목소리 샘플 2개 -> 평균 임베딩 DB 적재 + 캐시 무효화 (기존 아빠와 유사도 0 -> 정상 신규 등록)
    _set_stt(monkeypatch, SAMPLE_1)
    worker.process_voice_interaction(LONG_AUDIO)
    assert enrollment.state == EnrollmentState.WAITING_FOR_VOICE_SAMPLE
    _set_stt(monkeypatch, SAMPLE_2)
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
    submit_mock.assert_called_once_with("나 오늘 축구했어", user_id=new_user_id, speaker_ambiguous=False)


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
    monkeypatch.setattr(service, "preprocess", lambda audio, sample_rate=16000: audio)
    monkeypatch.setattr(service, "embed_preprocessed", lambda wav: np.array([-1.0, -1.0]))

    result = service.identify_speaker(LONG_AUDIO)

    assert result.soft_passed is True
    assert result.user_id == "user_new"
    assert result.display_name == "민수"
