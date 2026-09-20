"""
server/services/brain_service.py
Gemini 3.6 Flash 멀티모달 모델을 통한 감정 및 대사 추론 엔진.
"""
import re
import time
import logging
import warnings
from typing import Optional, List, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config import settings
from server.schemas.action import LLMResponse, EmotionType
from server.schemas.memory import MemoryConflictResult, MemoryRecord, MemoryRelation

logger = logging.getLogger(__name__)

DEFAULT_LLM_FALLBACK = LLMResponse(
    emotion=EmotionType.ERROR,
    speech="생각이 조금 엉켰어요. 잠시 후에 다시 말해줘!"
)


class BrainService:
    def __init__(self):
        self._client: Optional[genai.Client] = None
        self.system_instruction = (
            "너는 탁상형 반려로봇 Mentio의 두뇌 엔진이다. "
            "사용자의 발화와 전달된 스냅샷(있을 경우)을 종합해 친구처럼 다정하고 생동감 있게 반응하라.\n"
            "만약 사진이 전달되었더라도 사용자가 주변이나 물건을 보라고 한 게 아니라면, 사진을 굳이 언급하지 말고 자연스럽게 대화하라.\n\n"
            "[선택 가능한 감정]\n"
            "- HAPPY: 반가운 인사, 일상 대화, 칭찬, 기분 좋은 상황\n"
            "- PROUD: 사용자의 목표 달성 축하, 시험/업무 성공 격려, 멘티오의 자부심 표현\n"
            "- CURIOUS: 스냅샷 속 물건에 대해 물어볼 때, 사용자의 질문에 호기심을 보일 때\n"
            "- SAD: 속상한 일에 대한 공감, 아쉬움, 위로\n"
            "- ANGRY: 사용자의 짓궂은 장난에 투덜거리거나 귀엽게 삐칠 때\n"
            "- HEART_EYES: 스냅샷에서 애정 표현이 확인되었을 때, 고마움을 표현할 때\n"
            "- SURPRISED: 특이한 물체나 사용자의 뜻밖의 말에 깜짝 놀랐을 때\n"
            "- TIRED: 사용자가 피로를 호소하거나 멘티오가 함께 쉬자고 권유할 때\n\n"
            "반드시 영어 인사말, 생각(Thinking), 마크다운(```) 없이 오직 아래의 순수 단일 JSON 한 줄만 출력하라:\n"
            '{"emotion": "HAPPY", "speech": "안녕! 오늘 하루는 어땠어?"}'
        )
        # RAG 장기 기억 모순 판정(Invalidation) 전용 system instruction.
        # infer_action의 감정/대사 추론과는 무관한 별도 목적이므로 프롬프트를 분리한다.
        self.conflict_system_instruction = (
            "너는 로봇 Mentio의 장기 기억 저장소를 관리하는 판별기다.\n"
            "[신규 발화]가 [기존 기억 후보] 목록 중 어느 것과 사실적으로 모순되는지 판단하라.\n"
            "모순이란 같은 대상에 대한 선호/상태가 이전과 달라진 경우다 (예: '사과 좋아해' -> '사과 싫어해').\n"
            "단순히 주제가 비슷할 뿐 모순은 아닌 경우(예: '사과 좋아해'와 '포도 좋아해'는 서로 다른 대상이므로 "
            "모순이 아니다)는 반드시 NEW로 판단하라. 확실하지 않으면 NEW로 판단하라.\n"
            "반드시 마크다운, 설명 없이 오직 아래의 순수 단일 JSON 한 줄만 출력하라:\n"
            '{"relation": "CONTRADICTS", "conflicting_ids": [3]}\n'
            '모순이 없으면: {"relation": "NEW", "conflicting_ids": []}'
        )
        # [Memory Summarization] 활성 기억 목록 -> 고수준 페르소나 요약문 압축 전용 system instruction.
        # infer_action/classify_memory_relation과 무관한 별도 목적이므로 프롬프트를 분리한다.
        self.profile_summary_system_instruction = (
            "너는 로봇 Mentio가 사용자에 대해 낱개로 기억해 온 사실들을 종합해 "
            "하나의 자연스러운 한국어 페르소나 요약문으로 압축하는 역할이다.\n"
            "[기억 목록]에 나열된 사실들만 근거로 사용자의 성향/취향/생활 패턴을 알 수 있는 "
            "3~4문장 내외의 하나의 문단으로 요약하라.\n"
            "목록에 없는 내용을 추측하거나 지어내지 말고, 영어 인사말/마크다운/따옴표/JSON 없이 "
            "오직 순수 한국어 요약 문단 텍스트만 출력하라."
        )

    def get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=settings.GEMINI_API_KEY,
                http_options={"timeout": settings.API_TIMEOUT_MS}
            )
        return self._client

    @staticmethod
    def _extract_json_object(raw_text: str) -> Optional[str]:
        """
        모델 원시 출력에서 순수 JSON 객체 문자열만 뽑아낸다.
        마크다운 코드블록 제거 + 바깥쪽 중괄호 슬라이싱 + 잘린 문자열 자동 괄호 보정을 수행하며,
        parse_action_json과 classify_memory_relation이 이 로직을 공유한다.
        중괄호를 찾지 못하면 None을 반환한다 (⚠️ 이 auto-healing 로직 자체는 삭제 금지).
        """
        if not raw_text or not raw_text.strip():
            return None

        text = raw_text.strip()

        # 마크다운 블록 제거
        if "```" in text:
            text = re.sub(r"```(?:json)?", "", text)
            text = text.replace("```", "").strip()

        start_idx = text.find("{")
        end_idx = text.rfind("}")

        # 💡 [Auto-healing] speech 도중 문장이 잘려서 닫는 괄호가 없을 때 자동 복구
        if start_idx != -1 and (end_idx == -1 or end_idx <= start_idx):
            if not text.endswith('"'):
                text += '..."}'
            else:
                text += '}'
            end_idx = text.rfind("}")

        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            return text[start_idx : end_idx + 1]

        return None

    @staticmethod
    def parse_action_json(raw_text: str) -> LLMResponse:
        try:
            clean_json = BrainService._extract_json_object(raw_text)
            if clean_json is not None:
                return LLMResponse.model_validate_json(clean_json)

            if raw_text and raw_text.strip():
                logger.warning(f"[BrainService] 불완전한 JSON 구조 감지: {raw_text.strip()}")
            return DEFAULT_LLM_FALLBACK

        except Exception as e:
            logger.error(f"[BrainService Error] JSON 파싱 실패: {e} (Raw: {raw_text[:80]}...) -> Fallback 반환")
            return DEFAULT_LLM_FALLBACK

    def infer_action(self, contents: List[Any]) -> LLMResponse:
        client = self.get_client()

        # ⚡ 400 에러 방어: thinking_budget=0 대신 정식 지원 옵션 적용
        # (SDK 버전에 따라 thinking_budget을 지원하지 않거나 0을 거부하는 현상 차단)
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=1) # 0 대신 최소 단위인 1 지정
        )

        for attempt in range(settings.MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL_NAME,
                    contents=contents,
                    config=config
                )

                # 응답 텍스트 추출 (Parts 순회 백업 포함)
                raw_text = ""
                try:
                    raw_text = response.text or ""
                except Exception:
                    pass

                if not raw_text and response.candidates:
                    first_cand = response.candidates[0]
                    if first_cand.content and first_cand.content.parts:
                        raw_text = "".join([p.text for p in first_cand.content.parts if hasattr(p, "text") and p.text])

                if raw_text.strip():
                    return self.parse_action_json(raw_text)

                return DEFAULT_LLM_FALLBACK

            except APIError as e:
                if "503" in str(e) and attempt < settings.MAX_RETRIES:
                    logger.warning(f"[BrainService] 503 감지. 재시도 ({attempt + 1}/{settings.MAX_RETRIES})...")
                    time.sleep(1.0)
                    continue
                elif "429" in str(e):
                    logger.warning("[BrainService] 429 Quota Exceeded. 안전 탈출.")
                    return DEFAULT_LLM_FALLBACK
                else:
                    logger.error(f"[BrainService] API 오류: {e}")
                    return DEFAULT_LLM_FALLBACK
            except Exception as e:
                logger.error(f"[BrainService] 통신 오류: {e}")
                return DEFAULT_LLM_FALLBACK

        return DEFAULT_LLM_FALLBACK

    def classify_memory_relation(
        self, new_fact: str, candidates: List[MemoryRecord]
    ) -> MemoryConflictResult:
        """
        [RAG Invalidation] 저장 전 근접 중복(Dedup) 구간은 아니지만 주제가 겹치는 기존 기억
        후보들과 신규 발화 사이에 모순(선호/상태 변경)이 있는지 경량 LLM Reflection으로 판정한다.
        TTS 완료 후 백그라운드 스레드(memory_service.extract_and_store)에서만 호출되므로
        대화 턴 지연에는 영향이 없다. 후보가 없거나 호출 실패 시에는 기존 기억을 잘못 지우는
        것보다 안전한 쪽인 NEW(무효화 없음)로 폴백한다.
        """
        if not candidates:
            return MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[])

        candidate_lines = "\n".join(f'- id={c.id}: "{c.fact_text}"' for c in candidates)
        prompt = f'[신규 발화]\n"{new_fact}"\n\n[기존 기억 후보]\n{candidate_lines}'

        # ⚡ infer_action과 동일한 사유로 thinking_budget은 0이 아닌 1을 유지한다
        # (0 주입 시 400 INVALID_ARGUMENT, 미설정 시 기본 추론 프로세스로 인한 지연 발생).
        config = types.GenerateContentConfig(
            system_instruction=self.conflict_system_instruction,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=256,
            thinking_config=types.ThinkingConfig(thinking_budget=1),
        )

        try:
            client = self.get_client()
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL_NAME,
                contents=[prompt],
                config=config,
            )

            raw_text = ""
            try:
                raw_text = response.text or ""
            except Exception:
                pass

            if not raw_text and response.candidates:
                first_cand = response.candidates[0]
                if first_cand.content and first_cand.content.parts:
                    raw_text = "".join(
                        [p.text for p in first_cand.content.parts if hasattr(p, "text") and p.text]
                    )

            clean_json = self._extract_json_object(raw_text)
            if clean_json is None:
                logger.warning(f"[BrainService] 모순 판정 JSON 구조 불완전, NEW로 폴백: {raw_text[:80]}")
                return MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[])

            return MemoryConflictResult.model_validate_json(clean_json)

        except Exception as e:
            logger.warning(f"[BrainService] 기억 모순 판정 실패, 안전하게 NEW로 폴백: {e}")
            return MemoryConflictResult(relation=MemoryRelation.NEW, conflicting_ids=[])

    def summarize_profile(self, fact_texts: List[str]) -> str:
        """
        [Memory Summarization] 활성 기억 목록을 하나의 고수준 페르소나 요약문으로 압축한다.
        MemoryService.summarize_user_profile()이 대화 턴과 무관한 배치/백그라운드 파이프라인에서만
        호출하므로, infer_action(실시간 대화)만큼 엄격한 지연 요구사항은 없다.
        요약 결과는 정형 JSON이 아닌 순수 텍스트이므로 _extract_json_object 파서를 거치지 않는다
        (parse_action_json/classify_memory_relation 전용 Auto-healing 로직과는 별개 목적).
        """
        if not fact_texts:
            return ""

        facts_block = "\n".join(f"- {text}" for text in fact_texts)
        prompt = f"[기억 목록]\n{facts_block}"

        # ⚡ infer_action/classify_memory_relation과 동일한 사유로 thinking_budget은 0이 아닌 1을 유지한다
        # (0 주입 시 400 INVALID_ARGUMENT, 미설정 시 기본 추론 프로세스로 인한 지연 발생).
        config = types.GenerateContentConfig(
            system_instruction=self.profile_summary_system_instruction,
            temperature=0.3,
            max_output_tokens=1024,
            thinking_config=types.ThinkingConfig(thinking_budget=1),
        )

        try:
            client = self.get_client()
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL_NAME,
                contents=[prompt],
                config=config,
            )

            raw_text = ""
            try:
                raw_text = response.text or ""
            except Exception:
                pass

            if not raw_text and response.candidates:
                first_cand = response.candidates[0]
                if first_cand.content and first_cand.content.parts:
                    raw_text = "".join(
                        [p.text for p in first_cand.content.parts if hasattr(p, "text") and p.text]
                    )

            return raw_text.strip()

        except Exception as e:
            logger.warning(f"[BrainService] 프로필 요약 생성 실패: {e}")
            return ""


# Spring Bean 싱글톤 등록
brain_service = BrainService()