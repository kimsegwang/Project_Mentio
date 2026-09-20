"""
tests/test_brain_service.py
BrainService의 JSON 파싱(_extract_json_object/parse_action_json) 회귀 방지 및
RAG 장기 기억 모순 판정(classify_memory_relation)을 검증하는 pytest 단위 테스트.

검증 대상:
- _extract_json_object로 분리한 이후에도 parse_action_json의 기존 동작
  (순수 JSON 파싱, 마크다운 코드블록 제거, 잘린 문자열 auto-healing, 빈/불완전 입력 시 폴백)이
  회귀 없이 유지되는가
- classify_memory_relation은 후보가 없으면 LLM 호출 없이 즉시 NEW로 반환하는가
- classify_memory_relation은 thinking_budget=1 가드레일을 유지하는가
- classify_memory_relation은 응답 파싱 실패/예외 시 안전하게 NEW로 폴백하는가 (기존 기억 오삭제 방지)
"""
from unittest.mock import Mock

import pytest

from server.schemas.action import EmotionType, LLMResponse
from server.schemas.memory import MemoryConflictResult, MemoryRecord, MemoryRelation
from server.services.brain_service import BrainService, DEFAULT_LLM_FALLBACK


@pytest.fixture()
def service() -> BrainService:
    return BrainService()


# --- _extract_json_object / parse_action_json: 리팩터링 회귀 방지 ---

def test_parse_action_json_parses_pure_json(service):
    raw = '{"emotion": "HAPPY", "speech": "안녕! 오늘 하루는 어땠어?"}'
    result = service.parse_action_json(raw)

    assert result == LLMResponse(emotion=EmotionType.HAPPY, speech="안녕! 오늘 하루는 어땠어?")


def test_parse_action_json_strips_markdown_code_block(service):
    raw = '```json\n{"emotion": "SAD", "speech": "속상하겠다..."}\n```'
    result = service.parse_action_json(raw)

    assert result == LLMResponse(emotion=EmotionType.SAD, speech="속상하겠다...")


def test_parse_action_json_auto_heals_truncated_speech(service):
    # 닫는 중괄호 없이 문장 중간에 잘린 경우
    raw = '{"emotion": "HAPPY", "speech": "오늘 정말 좋은 하루였'
    result = service.parse_action_json(raw)

    assert result.emotion == EmotionType.HAPPY
    assert result.speech.startswith("오늘 정말 좋은 하루였")


def test_parse_action_json_returns_fallback_for_empty_text(service):
    assert service.parse_action_json("") == DEFAULT_LLM_FALLBACK
    assert service.parse_action_json("   ") == DEFAULT_LLM_FALLBACK


def test_parse_action_json_returns_fallback_when_no_braces_found(service):
    assert service.parse_action_json("그냥 인사말이야, JSON이 아님") == DEFAULT_LLM_FALLBACK


def test_extract_json_object_returns_none_without_braces(service):
    assert service._extract_json_object("no json here") is None


def test_extract_json_object_slices_outer_braces_only(service):
    raw = 'prefix noise { "a": 1 } trailing noise'
    assert service._extract_json_object(raw) == '{ "a": 1 }'


# --- classify_memory_relation(): 모순 판정 ---

def test_classify_memory_relation_skips_llm_call_when_no_candidates(service, monkeypatch):
    get_client_mock = Mock()
    monkeypatch.setattr(service, "get_client", get_client_mock)

    result = service.classify_memory_relation("사과 싫어해", [])

    assert result == MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[])
    get_client_mock.assert_not_called()


def test_classify_memory_relation_parses_contradicts_response(service, monkeypatch):
    candidate = MemoryRecord(id=5, user_id="primary_user", fact_text="사과를 좋아해", similarity=0.70, created_at=None)

    fake_response = Mock()
    fake_response.text = '{"relation": "CONTRADICTS", "conflicting_ids": [5]}'
    fake_response.candidates = []

    fake_client = Mock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(service, "get_client", lambda: fake_client)

    result = service.classify_memory_relation("이제 사과 싫고 포도 좋아", [candidate])

    assert result == MemoryConflictResult(relation=MemoryRelation.CONTRADICTS, conflicting_ids=[5])

    # 가드레일: thinking_budget은 반드시 1 (0이면 400 오류, 미설정 시 지연 발생)
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].thinking_config.thinking_budget == 1


def test_classify_memory_relation_parses_new_response(service, monkeypatch):
    candidate = MemoryRecord(id=7, user_id="primary_user", fact_text="포도를 좋아해", similarity=0.70, created_at=None)

    fake_response = Mock()
    fake_response.text = '{"relation": "NEW", "conflicting_ids": []}'
    fake_response.candidates = []

    fake_client = Mock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(service, "get_client", lambda: fake_client)

    result = service.classify_memory_relation("나는 사과를 좋아해", [candidate])

    assert result == MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[])


def test_classify_memory_relation_falls_back_to_new_on_unparseable_response(service, monkeypatch):
    candidate = MemoryRecord(id=5, user_id="primary_user", fact_text="사과를 좋아해", similarity=0.70, created_at=None)

    fake_response = Mock()
    fake_response.text = "이건 JSON이 아니야"
    fake_response.candidates = []

    fake_client = Mock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(service, "get_client", lambda: fake_client)

    result = service.classify_memory_relation("이제 사과 싫고 포도 좋아", [candidate])

    assert result == MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[])


def test_classify_memory_relation_falls_back_to_new_on_api_exception(service, monkeypatch):
    """기존 기억을 잘못 지우는 것보다 안전한 실패 모드(NEW, 무효화 없음)로 폴백해야 한다."""
    candidate = MemoryRecord(id=5, user_id="primary_user", fact_text="사과를 좋아해", similarity=0.70, created_at=None)

    def raise_error():
        raise RuntimeError("network down")

    monkeypatch.setattr(service, "get_client", raise_error)

    result = service.classify_memory_relation("이제 사과 싫고 포도 좋아", [candidate])

    assert result == MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[])


# --- summarize_profile(): [Memory Summarization] 활성 기억 -> 페르소나 요약 압축 ---

def test_summarize_profile_skips_llm_call_when_no_fact_texts(service, monkeypatch):
    get_client_mock = Mock()
    monkeypatch.setattr(service, "get_client", get_client_mock)

    result = service.summarize_profile([])

    assert result == ""
    get_client_mock.assert_not_called()


def test_summarize_profile_returns_response_text_and_keeps_thinking_budget(service, monkeypatch):
    fake_response = Mock()
    fake_response.text = "사용자는 커피를 좋아하고 아침형 인간이다."
    fake_response.candidates = []

    fake_client = Mock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(service, "get_client", lambda: fake_client)

    result = service.summarize_profile(["커피를 좋아한다", "매일 아침 산책한다"])

    assert result == "사용자는 커피를 좋아하고 아침형 인간이다."

    # 가드레일: thinking_budget은 반드시 1 (0이면 400 오류, 미설정 시 지연 발생)
    call_kwargs = fake_client.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].thinking_config.thinking_budget == 1
    assert call_kwargs["config"].max_output_tokens >= 1024

    prompt_sent = call_kwargs["contents"][0]
    assert "커피를 좋아한다" in prompt_sent
    assert "매일 아침 산책한다" in prompt_sent


def test_summarize_profile_falls_back_to_candidates_parts_when_text_empty(service, monkeypatch):
    fake_part = Mock()
    fake_part.text = "파츠 경로로 조립된 요약문"
    fake_content = Mock()
    fake_content.parts = [fake_part]
    fake_candidate = Mock()
    fake_candidate.content = fake_content

    fake_response = Mock()
    fake_response.text = ""
    fake_response.candidates = [fake_candidate]

    fake_client = Mock()
    fake_client.models.generate_content.return_value = fake_response
    monkeypatch.setattr(service, "get_client", lambda: fake_client)

    result = service.summarize_profile(["아무 사실"])

    assert result == "파츠 경로로 조립된 요약문"


def test_summarize_profile_returns_empty_string_on_api_exception(service, monkeypatch):
    def raise_error():
        raise RuntimeError("network down")

    monkeypatch.setattr(service, "get_client", raise_error)

    result = service.summarize_profile(["커피를 좋아한다"])

    assert result == ""
