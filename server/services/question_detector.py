"""
server/services/question_detector.py
RAG 검색 시 동적 임계값 완화 여부를 판별하기 위한 초경량 정규식 기반 의문문 감지기.

IntentService의 5단계 라우팅 파이프라인(호출어 정규화 -> 룰 기반 시간 -> 시각 부정어 veto ->
시각 요구 -> fallback)과는 별개의 관심사(RAG 임계값 조정)이므로 의도적으로 분리했다.
LLM 호출 없이 정규식만으로 판별해 지연을 추가하지 않는다.
"""
import re

# 의문형 종결 어미("~했지?", "~일까?") 및 의문사/회상 질의 키워드
QUESTION_PATTERNS = [
    r"(지|까|니|냐|나)\s*\??\s*$",  # 문장 끝 의문형 어미 (물음표는 STT 특성상 선택적)
    r"(뭐|무엇|무슨|누구|언제|어디|어떻게|왜|얼마)",  # 의문사
    r"기억\s*나",  # "기억나?", "기억나니" 등 과거 기억 회상 질의
]


class QuestionDetector:

  def __init__(self):
    self._patterns = [re.compile(p) for p in QUESTION_PATTERNS]

  def is_question(self, text: str) -> bool:
    if not text or not text.strip():
      return False

    cleaned = re.sub(r"[^\w\s?]", " ", text.strip())
    return any(p.search(cleaned) for p in self._patterns)


question_detector = QuestionDetector()
