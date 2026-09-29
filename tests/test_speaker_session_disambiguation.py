"""
tests/test_speaker_session_disambiguation.py
[Step 3] 화자 식별 안정화 검증 pytest 단위 테스트.

- 마진 검증: Top-1/Top-2 원점수 차이가 SPEAKER_MARGIN_THRESHOLD 미만이면 is_ambiguous + 2위 정보 반환
- 세션 락: 타임아웃 이내 세션 화자 가산/모호 시 우선권, 확실한 마진이면 세션 교체, 타임아웃 후 만료
- 호칭 정정: "나 {이름}이야" 계열 패턴 감지 -> 등록 화자 매칭 -> 세션 강제 전환 + 사과 멘트
- EMA: new = normalize((1-a)*current + a*input) 계산식, L2 norm=1 유지, 트리거 조건, 백그라운드 위임
- AIWorker/MemoryWriteWorker 연동: 모호 발화 기억 적재 억제, 정정 턴 처리, EMA 큐 위임

Resemblyzer 모델/DB는 전부 가짜로 대체한다. 기준 화자를 3차원 축 벡터로 두고 후보를 [a, b, c]로
구성하면 dad/mom 각각과의 코사인 유사도를 a, b로 정확히 지정할 수 있다.
"""
import time
from unittest.mock import Mock

import numpy as np
import pytest

import server.services.speaker_service as speaker_service_module
import server.workers.ai_worker as ai_worker_module
from server.repositories.speaker_repository import SpeakerEmbeddingRecord
from server.schemas.action import EmotionType, LLMResponse, TriggerType
from server.schemas.speaker import SpeakerCorrection, SpeakerIdentificationResult
from server.services.speaker_correction_service import SpeakerCorrectionService
from server.services.speaker_service import SpeakerService
from server.workers.ai_worker import AIWorker
from server.workers.memory_write_worker import MemoryWriteWorker

DAD = np.array([1.0, 0.0, 0.0])
MOM = np.array([0.0, 1.0, 0.0])
JIHUN = np.array([0.0, 0.0, 1.0])


def _candidate(sim_dad: float, sim_mom: float) -> np.ndarray:
    """dad/mom과 정확히 지정한 코사인 유사도를 갖는 단위 벡터 (나머지 성분은 JIHUN 축). sim_dad²+sim_mom² <= 1이어야 한다."""
    assert sim_dad**2 + sim_mom**2 <= 1.0
    rest = np.sqrt(max(0.0, 1.0 - sim_dad**2 - sim_mom**2))
    return np.array([sim_dad, sim_mom, rest])


def _records(include_jihun: bool = False):
    records = [
        SpeakerEmbeddingRecord(user_id="dad", display_name="아빠", embedding=DAD.tolist()),
        SpeakerEmbeddingRecord(user_id="mom", display_name="엄마", embedding=MOM.tolist()),
    ]
    if include_jihun:
        records.append(SpeakerEmbeddingRecord(user_id="jihun", display_name="지훈", embedding=JIHUN.tolist()))
    return records


@pytest.fixture()
def clock(monkeypatch):
    now = {"t": 1000.0}
    monkeypatch.setattr(speaker_service_module.time, "monotonic", lambda: now["t"])
    return now


@pytest.fixture()
def service(monkeypatch):
    s = speaker_service_module.settings
    monkeypatch.setattr(s, "SPEAKER_VERIFICATION_ENABLED", True)
    monkeypatch.setattr(s, "SPEAKER_SHORT_UTTERANCE_MAX_SEC", 1.0)
    monkeypatch.setattr(s, "SPEAKER_SESSION_SOFT_PASS_WINDOW_SEC", 10.0)
    monkeypatch.setattr(s, "SPEAKER_MARGIN_THRESHOLD", 0.04)
    monkeypatch.setattr(s, "SPEAKER_SESSION_TIMEOUT_SEC", 120.0)
    monkeypatch.setattr(s, "SPEAKER_SESSION_BONUS", 0.03)
    monkeypatch.setattr(s, "SPEAKER_EMA_ENABLED", True)
    monkeypatch.setattr(s, "SPEAKER_EMA_MIN_SIMILARITY", 0.82)
    monkeypatch.setattr(s, "SPEAKER_EMA_ALPHA", 0.05)
    monkeypatch.setattr(s, "SPEAKER_EMA_MIN_SPEECH_SEC", 1.0)
    svc = SpeakerService(similarity_threshold=0.65)
    monkeypatch.setattr(svc, "preprocess", lambda audio, sample_rate=16000: audio)
    return svc


def _patch_profiles(monkeypatch, records):
    monkeypatch.setattr(speaker_service_module, "get_active_speaker_embeddings", lambda: records)


def _speak(monkeypatch, service, embedding, seconds: float = 2.0):
    """임베딩 추론만 가짜로 대체해 식별한다. 전처리가 항등이므로 오디오 길이 = 실제 발화 길이."""
    monkeypatch.setattr(service, "embed_preprocessed", lambda wav: np.asarray(embedding, dtype=np.float32))
    return service.identify_speaker(np.zeros(int(16000 * seconds), dtype=np.float32))


# --- 1. 마진 검증 ---

def test_margin_below_threshold_returns_ambiguous_with_top2_info(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records())

    result = _speak(monkeypatch, service, _candidate(0.70, 0.68))

    assert result.is_match is True
    assert result.user_id == "dad"
    assert result.is_ambiguous is True
    assert result.margin == pytest.approx(0.02, abs=1e-4)
    assert result.second_user_id == "mom"
    assert result.second_display_name == "엄마"
    assert result.second_similarity == pytest.approx(0.68, abs=1e-4)


def test_margin_above_threshold_is_not_ambiguous(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records())

    result = _speak(monkeypatch, service, _candidate(0.70, 0.60))

    assert result.user_id == "dad"
    assert result.is_ambiguous is False
    assert result.margin == pytest.approx(0.10, abs=1e-4)


def test_single_candidate_is_never_ambiguous(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records()[:1])

    result = _speak(monkeypatch, service, _candidate(0.80, 0.0))

    assert result.is_ambiguous is False
    assert result.margin is None
    assert result.second_user_id is None


# --- 2. 세션 락 ---

def test_session_bonus_applies_within_timeout(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records())
    assert _speak(monkeypatch, service, _candidate(0.90, 0.10)).user_id == "dad"

    # 소프트패스 창(10초)은 지났지만 세션(120초) 이내: 원점수 0.63 + 가산 0.03 = 0.66 >= 0.65
    clock["t"] += 60.0
    result = _speak(monkeypatch, service, _candidate(0.63, 0.10))

    assert result.is_match is True
    assert result.user_id == "dad"
    assert result.soft_passed is False
    assert result.session_bonus_applied is True
    assert result.similarity == pytest.approx(0.63, abs=1e-4)  # 보고 점수는 가산 전 원점수


def test_session_expires_after_timeout(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records())
    _speak(monkeypatch, service, _candidate(0.90, 0.10))
    assert service.get_session_speaker() == ("dad", "아빠")

    clock["t"] += 121.0
    assert service.get_session_speaker() is None
    result = _speak(monkeypatch, service, _candidate(0.63, 0.10))

    assert result.is_match is False
    assert result.session_bonus_applied is False


def test_ambiguous_margin_gives_priority_to_session_speaker(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records())
    _speak(monkeypatch, service, _candidate(0.10, 0.90))  # 세션 화자: mom

    clock["t"] += 30.0
    result = _speak(monkeypatch, service, _candidate(0.70, 0.68))  # dad가 근소하게 1위 (0.70 vs 0.68)

    assert result.is_ambiguous is True
    assert result.user_id == "mom"
    assert result.resolved_by_session is True
    assert result.session_bonus_applied is True
    assert result.similarity == pytest.approx(0.68, abs=1e-4)


def test_clear_margin_switches_session_speaker(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records())
    _speak(monkeypatch, service, _candidate(0.10, 0.90))  # 세션 화자: mom

    clock["t"] += 30.0
    result = _speak(monkeypatch, service, _candidate(0.72, 0.62))  # dad가 확실히 앞섬 (0.72 vs 0.62)

    assert result.is_ambiguous is False
    assert result.user_id == "dad"
    assert result.resolved_by_session is False
    assert service.get_session_speaker() == ("dad", "아빠")


# --- 3. 호칭 정정 ---

@pytest.mark.parametrize(
    "text, expected",
    [
        ("나 지훈이야", "지훈"),
        ("나 민수인데?", "민수"),
        ("나 민수야", "민수"),
        ("멘티오, 나는 엄마야!", "엄마"),
        ("저 지훈이에요", "지훈"),
        ("나 민수 아니고 지훈이야", "지훈"),
        ("민수가 아니라 지훈이라니까", "지훈"),
    ],
)
def test_extract_claimed_name_detects_correction_patterns(text, expected):
    correction_service = SpeakerCorrectionService(speaker_service_instance=Mock())
    assert correction_service.extract_claimed_name(text) in (expected, expected + "이")


@pytest.mark.parametrize("text", ["나 오늘 축구했어", "나 민수 아니야", "나 민수인데 오늘 축구했어", "민수야 안녕"])
def test_extract_claimed_name_ignores_non_correction_utterances(text):
    correction_service = SpeakerCorrectionService(speaker_service_instance=Mock())
    assert correction_service.extract_claimed_name(text) is None


def test_correction_matches_registered_speaker_and_forces_session_switch(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records(include_jihun=True))
    _speak(monkeypatch, service, _candidate(0.10, 0.90))  # mom으로 (잘못) 식별
    correction_service = SpeakerCorrectionService(speaker_service_instance=service)

    correction = correction_service.detect("나 지훈이야", current_user_id="mom")

    assert correction == SpeakerCorrection(
        user_id="jihun", display_name="지훈", spoken_name=correction.spoken_name, previous_user_id="mom"
    )
    speech = correction_service.apply(correction)
    assert speech == "앗, 지훈아 미안해! 목소리가 비슷해서 착각했어."
    assert service.get_session_speaker() == ("jihun", "지훈")


def test_correction_ignored_when_name_matches_current_or_unregistered(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records(include_jihun=True))
    correction_service = SpeakerCorrectionService(speaker_service_instance=service)

    assert correction_service.detect("나 지훈이야", current_user_id="jihun") is None  # 이미 맞게 식별
    assert correction_service.detect("나 학생인데", current_user_id="mom") is None  # 미등록 이름


@pytest.mark.parametrize("name, expected", [("지훈", "지훈아"), ("민수", "민수야"), ("Tom", "Tom,")])
def test_vocative_particle(name, expected):
    assert SpeakerCorrectionService.vocative(name) == expected


# --- 4. EMA ---

def test_ema_formula_and_l2_normalization():
    current = np.array([1.0, 0.0, 0.0])
    new_input = np.array([0.0, 1.0, 0.0])

    updated = SpeakerService.compute_ema_embedding(current, new_input, alpha=0.05)

    expected = np.array([0.95, 0.05, 0.0])
    expected /= np.linalg.norm(expected)
    assert np.allclose(updated, expected, atol=1e-6)
    assert np.linalg.norm(updated) == pytest.approx(1.0, abs=1e-6)


def test_ema_keeps_unit_norm_over_repeated_updates():
    rng = np.random.default_rng(0)
    vector = SpeakerService.average_embeddings([rng.normal(size=256)])
    for _ in range(50):
        vector = SpeakerService.compute_ema_embedding(vector, SpeakerService.average_embeddings([rng.normal(size=256)]), 0.05)
    assert np.linalg.norm(vector) == pytest.approx(1.0, abs=1e-5)


def _eligible_result(**overrides):
    base = dict(
        is_match=True, user_id="dad", similarity=0.90, threshold=0.65,
        speech_duration_sec=2.0, embedding=[1.0, 0.0, 0.0],
    )
    base.update(overrides)
    return SpeakerIdentificationResult(**base)


@pytest.mark.parametrize(
    "overrides, eligible",
    [
        ({}, True),
        ({"similarity": 0.81}, False),
        ({"is_ambiguous": True}, False),
        ({"speech_duration_sec": 0.9}, False),
        ({"soft_passed": True}, False),
        ({"skipped": True}, False),
        ({"embedding": None}, False),
    ],
)
def test_ema_eligibility_conditions(service, overrides, eligible):
    assert SpeakerService.is_ema_eligible(_eligible_result(**overrides)) is eligible


def test_ema_disabled_flag_blocks_update(service, monkeypatch):
    monkeypatch.setattr(speaker_service_module.settings, "SPEAKER_EMA_ENABLED", False)
    assert SpeakerService.is_ema_eligible(_eligible_result()) is False


def test_identify_result_carries_embedding_and_duration_for_ema(service, monkeypatch, clock):
    _patch_profiles(monkeypatch, _records())

    result = _speak(monkeypatch, service, _candidate(0.90, 0.10), seconds=1.5)

    assert result.speech_duration_sec == pytest.approx(1.5)
    assert np.allclose(result.embedding, _candidate(0.90, 0.10), atol=1e-6)
    assert "embedding" not in result.model_dump()  # 로그/직렬화에는 노출되지 않음
    assert SpeakerService.is_ema_eligible(result) is True


def test_apply_ema_update_reads_db_vector_and_writes_normalized_result(service, monkeypatch):
    monkeypatch.setattr(speaker_service_module, "get_speaker_embedding", lambda user_id: [1.0, 0.0, 0.0])
    written = {}

    def fake_update(user_id, embedding):
        written[user_id] = embedding
        return True

    monkeypatch.setattr(speaker_service_module, "update_speaker_embedding", fake_update)

    assert service.apply_ema_update("dad", [0.0, 1.0, 0.0]) is True
    assert np.linalg.norm(written["dad"]) == pytest.approx(1.0, abs=1e-6)
    assert written["dad"][0] > written["dad"][1] > 0.0


def test_apply_ema_update_skips_when_profile_missing(service, monkeypatch):
    monkeypatch.setattr(speaker_service_module, "get_speaker_embedding", lambda user_id: None)
    update_mock = Mock()
    monkeypatch.setattr(speaker_service_module, "update_speaker_embedding", update_mock)

    assert service.apply_ema_update("ghost", [0.0, 1.0, 0.0]) is False
    update_mock.assert_not_called()


# --- MemoryWriteWorker 연동 ---

def _wait_until(predicate, timeout=2.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_memory_worker_suppresses_ambiguous_speaker_writes(monkeypatch):
    import server.workers.memory_write_worker as worker_module

    extract_mock = Mock()
    monkeypatch.setattr(worker_module.memory_service, "extract_and_store", extract_mock)
    worker = MemoryWriteWorker(speaker_service_instance=Mock())
    worker.start()
    try:
        worker.submit("나 축구 좋아해", user_id="dad", speaker_ambiguous=True)
        assert _wait_until(lambda: worker._queue.unfinished_tasks == 0)
        time.sleep(0.05)
        extract_mock.assert_not_called()
    finally:
        worker.stop()


def test_memory_worker_runs_speaker_ema_task_in_background(monkeypatch):
    fake_speaker_service = Mock()
    worker = MemoryWriteWorker(speaker_service_instance=fake_speaker_service)
    worker.start()
    try:
        worker.submit_speaker_ema("dad", [1.0, 0.0, 0.0])
        assert _wait_until(lambda: fake_speaker_service.apply_ema_update.called)
        fake_speaker_service.apply_ema_update.assert_called_once_with("dad", [1.0, 0.0, 0.0])
    finally:
        worker.stop()


# --- AIWorker 파이프라인 연동 ---

@pytest.fixture()
def ai_worker(monkeypatch):
    mock_brain = Mock()
    mock_brain.infer_action.return_value = LLMResponse(emotion=EmotionType.HAPPY, speech="테스트 응답")
    mock_tts = Mock()
    mock_tts.synthesize.return_value = b""
    w = AIWorker(
        brain_service_instance=mock_brain,
        tts_service_instance=mock_tts,
        audio_player_instance=Mock(),
        correction_service_instance=Mock(),
    )
    w.correction_service.detect.return_value = None
    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", Mock())
    monkeypatch.setattr(ai_worker_module.intent_service, "analyze_voice_intent", lambda text: (TriggerType.VOICE_CHAT.value, False))
    monkeypatch.setattr(w, "_retrieve_memory_context", lambda user_text, user_id: "")
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", Mock())
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit_speaker_ema", Mock())
    return w


def _identify_as(monkeypatch, result: SpeakerIdentificationResult):
    monkeypatch.setattr(ai_worker_module.speaker_service, "identify_speaker", lambda *args, **kwargs: result)


def test_ai_worker_suppresses_memory_write_when_speaker_ambiguous(ai_worker, monkeypatch):
    _identify_as(monkeypatch, _eligible_result(
        similarity=0.80, is_ambiguous=True, margin=0.02, second_user_id="mom", second_similarity=0.78
    ))
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "나 축구 좋아해")

    ai_worker.process_voice_interaction(audio_data=b"dummy")

    ai_worker_module.memory_write_worker.submit.assert_called_once_with(
        "나 축구 좋아해", user_id="dad", speaker_ambiguous=True
    )
    ai_worker_module.memory_write_worker.submit_speaker_ema.assert_not_called()


def test_ai_worker_submits_ema_for_high_confidence_utterance(ai_worker, monkeypatch):
    _identify_as(monkeypatch, _eligible_result())
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "오늘 날씨 좋다")

    ai_worker.process_voice_interaction(audio_data=b"dummy")

    ai_worker_module.memory_write_worker.submit_speaker_ema.assert_called_once_with("dad", [1.0, 0.0, 0.0])


def test_ai_worker_correction_turn_switches_speaker_without_llm_or_memory(ai_worker, monkeypatch):
    _identify_as(monkeypatch, _eligible_result(user_id="mom"))
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "나 지훈이야")
    correction = SpeakerCorrection(user_id="jihun", display_name="지훈", spoken_name="지훈", previous_user_id="mom")
    ai_worker.correction_service.detect.return_value = correction
    ai_worker.correction_service.apply.return_value = "앗, 지훈아 미안해! 목소리가 비슷해서 착각했어."

    action = ai_worker.process_voice_interaction(audio_data=b"dummy")

    ai_worker.correction_service.detect.assert_called_once_with("나 지훈이야", current_user_id="mom")
    ai_worker.correction_service.apply.assert_called_once_with(correction)
    assert action.speech == "앗, 지훈아 미안해! 목소리가 비슷해서 착각했어."
    _, trigger, _ = ai_worker.response_queue.get_nowait()
    assert trigger == TriggerType.VOICE_SPEAKER_CORRECTION.value
    ai_worker.brain_service.infer_action.assert_not_called()
    ai_worker_module.memory_write_worker.submit.assert_not_called()
    ai_worker_module.memory_write_worker.submit_speaker_ema.assert_not_called()
