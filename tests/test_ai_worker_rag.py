"""
tests/test_ai_worker_rag.py
AIWorker에 통합된 RAG 파이프라인(A-1: Retrieval 훅 / B-1: 백그라운드 적재 훅)을
검증하는 pytest 단위 테스트.

설계 결정 검증 대상:
- VOICE_TIME_RULE일 때는 RAG 검색(A-1)과 백그라운드 적재(B-1) 모두 스킵
- 그 외(VOICE_CHAT/VOICE_VISION)에는 STT 완료 후 Top-K 기억을
  "[참고 기억] ..." 형태로 Gemini contents에 주입
- TTS 완료 후 규칙 기반 필터를 거쳐 백그라운드 스레드로 비동기 적재 시도
"""
from unittest.mock import Mock

import pytest

import server.workers.ai_worker as ai_worker_module
from server.schemas.action import EmotionType, LLMResponse, TriggerType
from server.schemas.memory import MemoryRecord
from server.workers.ai_worker import AIWorker


class ImmediateThread:
    """threading.Thread를 대체하는 동기 실행 테스트 더블 (fire-and-forget을 결정적으로 검증)"""

    def __init__(self, target=None, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}

    def start(self):
        self._target(*self._args, **self._kwargs)


@pytest.fixture()
def worker(monkeypatch) -> AIWorker:
    # DB/TTS/오디오/로그 등 실제 I/O를 전부 목(Mock)으로 대체한 순수 단위 테스트용 워커
    mock_brain = Mock()
    mock_brain.infer_action.return_value = LLMResponse(emotion=EmotionType.HAPPY, speech="테스트 응답")
    mock_tts = Mock()
    mock_tts.synthesize.return_value = b""
    mock_audio_player = Mock()

    w = AIWorker(
        brain_service_instance=mock_brain,
        tts_service_instance=mock_tts,
        audio_player_instance=mock_audio_player,
    )

    monkeypatch.setattr(ai_worker_module, "insert_interaction_log", Mock())
    monkeypatch.setattr(ai_worker_module.threading, "Thread", ImmediateThread)
    return w


# --- _retrieve_memory_context(): Top-K 컨텍스트 조립 ---

def test_retrieve_memory_context_formats_topk_as_reference_lines(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", lambda text: [0.1] * 384)
    monkeypatch.setattr(
        ai_worker_module,
        "search_similar_memories",
        lambda embedding, top_k, threshold: [
            MemoryRecord(id=1, user_id="primary_user", fact_text="커피를 좋아한다", similarity=0.9),
            MemoryRecord(id=2, user_id="primary_user", fact_text="매일 아침 산책한다", similarity=0.8),
        ],
    )

    context = worker._retrieve_memory_context("오늘 뭐 마실까?")

    assert context == "[참고 기억] 커피를 좋아한다\n[참고 기억] 매일 아침 산책한다"


def test_retrieve_memory_context_returns_empty_when_no_hits(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", lambda text: [0.1] * 384)
    monkeypatch.setattr(
        ai_worker_module, "search_similar_memories", lambda embedding, top_k, threshold: []
    )

    assert worker._retrieve_memory_context("아무 상관 없는 발화") == ""


def test_retrieve_memory_context_prints_similarity_score_per_memory(worker, monkeypatch, capsys):
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", lambda text: [0.1] * 384)
    monkeypatch.setattr(
        ai_worker_module,
        "search_similar_memories",
        lambda embedding, top_k, threshold: [
            MemoryRecord(id=1, user_id="primary_user", fact_text="커피를 좋아한다", similarity=0.912345),
        ],
    )

    context = worker._retrieve_memory_context("오늘 뭐 마실까?")

    # 실제 Gemini 프롬프트에 주입되는 컨텍스트 문자열 자체는 기존 포맷 그대로 유지
    assert context == "[참고 기억] 커피를 좋아한다"
    # 콘솔에 항상 보이도록 print로 유사도 점수가 소수점 둘째 자리까지 함께 찍힌다
    captured = capsys.readouterr()
    assert "🧠 [RAG] [참고 기억] 커피를 좋아한다 (유사도: 0.91)" in captured.out


def test_retrieve_memory_context_prints_skip_message_when_no_hits(worker, monkeypatch, capsys):
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", lambda text: [0.1] * 384)
    monkeypatch.setattr(
        ai_worker_module, "search_similar_memories", lambda embedding, top_k, threshold: []
    )

    worker._retrieve_memory_context("아무 상관 없는 발화")

    captured = capsys.readouterr()
    assert "🧠 [RAG]" in captured.out
    assert "생략" in captured.out


# --- 의문문 감지 시 동적 임계값(RAG_QUESTION_SIMILARITY_THRESHOLD) 적용 ---

def test_retrieve_memory_context_uses_relaxed_threshold_for_question(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", lambda text: [0.1] * 384)

    captured_kwargs = {}

    def fake_search(embedding, top_k, threshold):
        captured_kwargs["threshold"] = threshold
        return []

    monkeypatch.setattr(ai_worker_module, "search_similar_memories", fake_search)

    worker._retrieve_memory_context("내가 무슨 과일 좋아한다고 했지?")

    assert captured_kwargs["threshold"] == ai_worker_module.settings.RAG_QUESTION_SIMILARITY_THRESHOLD


def test_retrieve_memory_context_uses_default_threshold_for_statement(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", lambda text: [0.1] * 384)

    captured_kwargs = {}

    def fake_search(embedding, top_k, threshold):
        captured_kwargs["threshold"] = threshold
        return []

    monkeypatch.setattr(ai_worker_module, "search_similar_memories", fake_search)

    worker._retrieve_memory_context("난 포도 좋아")

    assert captured_kwargs["threshold"] == ai_worker_module.settings.RAG_SIMILARITY_THRESHOLD


def test_retrieve_memory_context_logs_relaxed_threshold_for_question(worker, monkeypatch, capsys):
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", lambda text: [0.1] * 384)
    monkeypatch.setattr(
        ai_worker_module, "search_similar_memories", lambda embedding, top_k, threshold: []
    )

    worker._retrieve_memory_context("내가 무슨 과일 좋아한다고 했지?")

    captured = capsys.readouterr()
    assert "의문문 감지" in captured.out
    assert str(ai_worker_module.settings.RAG_QUESTION_SIMILARITY_THRESHOLD) in captured.out


def test_retrieve_memory_context_returns_empty_on_failure_without_raising(worker, monkeypatch):
    def raise_error(text):
        raise RuntimeError("embedding backend down")

    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", raise_error)

    assert worker._retrieve_memory_context("아무 발화") == ""


# --- process_voice_interaction(): 파이프라인 통합 훅 ---

def test_time_rule_skips_rag_retrieval_and_background_storage(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "지금 몇 시야")
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "analyze_voice_intent",
        lambda text: (TriggerType.VOICE_TIME_RULE.value, False),
    )
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "build_time_response",
        lambda: LLMResponse(emotion=EmotionType.HAPPY, speech="지금은 오후 3시야!"),
    )

    search_mock = Mock()
    embed_mock = Mock()
    submit_mock = Mock()
    monkeypatch.setattr(ai_worker_module, "search_similar_memories", search_mock)
    monkeypatch.setattr(ai_worker_module.embedding_service, "embed", embed_mock)
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", submit_mock)

    action = worker.process_voice_interaction(audio_data=b"dummy")

    assert action.speech == "지금은 오후 3시야!"
    search_mock.assert_not_called()
    embed_mock.assert_not_called()
    submit_mock.assert_not_called()


def test_voice_chat_injects_memory_context_into_gemini_contents(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "오늘 기분이 좋아")
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "analyze_voice_intent",
        lambda text: (TriggerType.VOICE_CHAT.value, False),
    )
    monkeypatch.setattr(
        worker,
        "_retrieve_memory_context",
        lambda user_text: "[참고 기억] 사용자는 커피를 좋아한다",
    )

    worker.process_voice_interaction(audio_data=b"dummy")

    contents_passed = worker.brain_service.infer_action.call_args[0][0]
    assert contents_passed[0] == "[참고 기억] 사용자는 커피를 좋아한다"
    assert contents_passed[-1] == '사용자의 음성 대화: "오늘 기분이 좋아"'


def test_voice_chat_triggers_background_storage_after_tts(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "내 이름은 김세강이야")
    monkeypatch.setattr(
        ai_worker_module.intent_service,
        "analyze_voice_intent",
        lambda text: (TriggerType.VOICE_CHAT.value, False),
    )
    monkeypatch.setattr(worker, "_retrieve_memory_context", lambda user_text: "")

    submit_mock = Mock()
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", submit_mock)

    worker.process_voice_interaction(audio_data=b"dummy")

    # MemoryWriteWorker의 순차 큐에 위임(submit)만 하고 즉시 반환되므로 대화 턴에는 지연이 없다.
    # 실제 extract_and_store 실행은 MemoryWriteWorker 자체의 단위 테스트에서 검증한다.
    submit_mock.assert_called_once_with("내 이름은 김세강이야")


def test_no_recognized_text_does_not_touch_rag_pipeline(worker, monkeypatch):
    monkeypatch.setattr(ai_worker_module.stt_service, "transcribe", lambda audio: "")

    search_mock = Mock()
    submit_mock = Mock()
    monkeypatch.setattr(ai_worker_module, "search_similar_memories", search_mock)
    monkeypatch.setattr(ai_worker_module.memory_write_worker, "submit", submit_mock)

    result = worker.process_voice_interaction(audio_data=b"dummy")

    assert result is None
    search_mock.assert_not_called()
    submit_mock.assert_not_called()
