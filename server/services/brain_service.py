import os
import re
import time
import logging
from typing import Optional, List, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config import settings
from server.schemas.action import LLMResponse, EmotionType

logger = logging.getLogger(__name__)

# LLM 레벨 파싱 실패 시 기본 Fallback 응답
DEFAULT_LLM_FALLBACK = LLMResponse(
    emotion=EmotionType.ERROR,
    speech="생각이 조금 엉켰어요. 잠시 후에 다시 말해줘!"
)


class BrainService:
    def __init__(self):
        self._client: Optional[genai.Client] = None
        self.system_instruction = (
            "너는 탁상형 반려로봇 Mentio의 두뇌 엔진이다. "
            "사용자가 건넨 말(텍스트)과 전달된 스냅샷 이미지를 종합해 친구처럼 다정하고 생동감 있게 반응하라.\n\n"
            "[선택 가능한 감정]\n"
            "- HAPPY: 반가운 인사, 일상 대화, 칭찬, 기분 좋은 상황\n"
            "- PROUD: 사용자의 목표 달성 축하, 시험/업무 성공 격려, 멘티오의 자부심 표현\n"
            "- CURIOUS: 스냅샷 속 물건에 대해 물어볼 때, 사용자의 질문에 호기심을 보일 때\n"
            "- SAD: 속상한 일에 대한 공감, 아쉬움, 위로\n"
            "- ANGRY: 사용자의 짓궂은 장난에 투덜거리거나 귀엽게 삐칠 때\n"
            "- HEART_EYES: 스냅샷에서 양손 하트나 애정 표현이 확인되었을 때, 고마움을 표현할 때\n"
            "- SURPRISED: 스냅샷 속 특이한 물체나 사용자의 뜻밖의 말에 깜짝 놀랐을 때\n"
            "- TIRED: 사용자가 피로/지침을 호소하거나 멘티오가 함께 쉬자고 권유할 때\n\n"
            "규칙: 반드시 위 8가지 감정 중 하나를 emotion으로 선택하고, 1~2문장의 자연스러운 한국어 구어체 speech를 생성하라."
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
        """사족 텍스트 제거 및 Pydantic LLMResponse 객체 파싱"""
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                return LLMResponse.model_validate_json(json_match.group(0))
            return LLMResponse.model_validate_json(cleaned)
        except Exception as e:
            logger.error(f"[BrainService Error] JSON 파싱 실패: {e} -> Fallback 반환")
            return DEFAULT_LLM_FALLBACK

    def infer_action(self, contents: List[Any]) -> LLMResponse:
        """Gemini API 호출 및 예외(503, 429, Timeout) 방어"""
        client = self.get_client()
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            response_mime_type="application/json",
            response_schema=LLMResponse,
            temperature=0.7
        )

        for attempt in range(settings.MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=settings.GEMINI_MODEL_NAME,
                    contents=contents,
                    config=config
                )
                if response.text:
                    return self.parse_action_json(response.text)
                return DEFAULT_LLM_FALLBACK

            except APIError as e:
                if "503" in str(e) and attempt < settings.MAX_RETRIES:
                    logger.warning(f"[BrainService] 503 과부하 감지. 1초 대기 후 재시도 ({attempt + 1}/{settings.MAX_RETRIES})...")
                    time.sleep(1.0)
                    continue
                elif "429" in str(e):
                    logger.warning("[BrainService] 429 Quota Exceeded 감지. 안전 탈출합니다.")
                    return DEFAULT_LLM_FALLBACK
                else:
                    logger.error(f"[BrainService] API 오류 발생: {e}")
                    return DEFAULT_LLM_FALLBACK
            except Exception as e:
                logger.error(f"[BrainService] 예상치 못한 통신 오류: {e}")
                return DEFAULT_LLM_FALLBACK

        return DEFAULT_LLM_FALLBACK

    def classify_vision_intent(self, text: str) -> bool:
        """
        텍스트 문맥을 분석하여 눈앞의 시각 정보(카메라 스냅샷)가 필요한지 판별합니다.
        (Tier 2 Semantic Routing)
        """
        prompt = (
            "당신은 탁상형 반려로봇의 의도 분석기입니다. "
            "사용자의 발화 문맥을 보고, 로봇이 '지금 눈앞의 사물/사람/주변 환경을 카메라로 직접 보아야 하는 상황'인지 판단하세요.\n\n"
            "판단 기준:\n"
            "- 로봇에게 눈앞의 무언가를 보거나, 사진을 찍거나, 표정/옷/사물을 확인해달라고 하면: YES\n"
            "- 과거의 일, 단순 지식 질문, 일반 일상 대화, 보지 않아도 대답할 수 있으면: NO\n\n"
            f"사용자 발화: \"{text}\"\n"
            "답변: (YES 또는 NO만 한 단어로 출력)"
        )
        try:
            client = self.get_client()
            response = client.models.generate_content(
                model=settings.GEMINI_MODEL_NAME,
                contents=prompt
            )
            result = response.text.strip().upper()
            return "YES" in result
        except Exception as e:
            logger.error(f"Failed to classify vision intent with LLM: {e}")
            return False


# Spring Bean 싱글톤 인스턴스 생성 (외부 import용)
brain_service = BrainService()