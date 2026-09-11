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

    def get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=settings.GEMINI_API_KEY,
                http_options={"timeout": settings.API_TIMEOUT_MS}
            )
        return self._client

    @staticmethod
    def parse_action_json(raw_text: str) -> LLMResponse:
        try:
            if not raw_text or not raw_text.strip():
                return DEFAULT_LLM_FALLBACK

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
                clean_json = text[start_idx : end_idx + 1]
                return LLMResponse.model_validate_json(clean_json)

            logger.warning(f"[BrainService] 불완전한 JSON 구조 감지: {text}")
            return DEFAULT_LLM_FALLBACK

        except Exception as e:
            logger.error(f"[BrainService Error] JSON 파싱 실패: {e} (Raw: {raw_text[:80]}...) -> Fallback 반환")
            return DEFAULT_LLM_FALLBACK

    def infer_action(self, contents: List[Any]) -> LLMResponse:
        client = self.get_client()

        # types 객체 충돌 없이 순수 JSON 강제 + 1024 토큰 설정
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            response_mime_type="application/json",
            temperature=0.2,
            max_output_tokens=1024
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

# Spring Bean 싱글톤 등록
brain_service = BrainService()