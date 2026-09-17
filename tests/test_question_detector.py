"""
tests/test_question_detector.py
QuestionDetector(server/services/question_detector.py)의 의문문 판별 정규식을 검증하는 pytest 테스트.

검증 대상:
- RAG 임계값 동적 완화 분기(ai_worker._retrieve_memory_context)에 넘길 is_question 플래그가
  실제 서술문/의문문을 정확히 구분하는지
- LLM 호출 없이 정규식만으로 판별되므로 예외 없이 항상 bool을 반환하는지 (빈 문자열 등 엣지 케이스 포함)
"""
import pytest

from server.services.question_detector import QuestionDetector


@pytest.fixture()
def detector() -> QuestionDetector:
    return QuestionDetector()


# --- 의문문으로 판별되어야 하는 발화 ---

@pytest.mark.parametrize(
    "text",
    [
        "내가 무슨 과일 좋아한다고 했지?",
        "내가 무슨 과일 좋아한다고 했지",  # STT가 물음표를 누락해도 어미로 판별
        "그거 기억나?",
        "기억나니",
        "내가 어디 산다고 했지",
        "내 생일이 언제라고 했어",
        "나 누구 좋아한다고 했지",
        "이번 주말에 뭐 한다고 했었지?",
        "내가 그거 얼마라고 했지",
        "커피 좋아한다고 했었나",
    ],
)
def test_detects_question_utterances(detector: QuestionDetector, text: str):
    assert detector.is_question(text) is True


# --- 서술문/일반 발화는 의문문으로 오탐하면 안 됨 ---

@pytest.mark.parametrize(
    "text",
    [
        "난 포도 좋아",
        "오늘 기분이 좋아",
        "내 이름은 김세강이야",
        "매일 아침 산책한다",
        "커피를 좋아한다",
    ],
)
def test_does_not_flag_statements_as_questions(detector: QuestionDetector, text: str):
    assert detector.is_question(text) is False


# --- 엣지 케이스: 빈 문자열/공백은 예외 없이 False ---

@pytest.mark.parametrize("text", ["", "   ", None])
def test_empty_or_none_text_is_not_a_question(detector: QuestionDetector, text):
    assert detector.is_question(text) is False
