import os
import re
import time
from typing import Optional, List, Any
from google import genai
from google.genai import types
from google.genai.errors import APIError

from config import settings
from server.schemas.action import RobotAction, FALLBACK_ACTION


class BrainService:
    def __init__(self):
        self._client: Optional[genai.Client] = None
        self.system_instruction = (
            "너는 탁상형 반려로봇 Mentio의 두뇌 엔진이다. "
            "주어진 시각/제스처 상황을 파악하고 친구처럼 다정하게 반응하라. "
            "반드시 정의된 JSON 스키마 규격으로만 응답해야 한다."
        )

    def get_client(self) -> genai.Client:
        if self._client is None:
            self._client = genai.Client(
                api_key=settings.GEMINI_API_KEY,
                http_options={"timeout": settings.API_TIMEOUT_MS}
            )
        return self._client

    @staticmethod
    def parse_action_json(raw_text: str) -> RobotAction:
        """사족 텍스트 제거 및 Pydantic RobotAction 객체 파싱"""
        try:
            cleaned = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.MULTILINE)
            cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
            json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if json_match:
                return RobotAction.model_validate_json(json_match.group(0))
            return RobotAction.model_validate_json(cleaned)
        except Exception as e:
            print(f"[BrainService Error] JSON 파싱 실패: {e} -> Fallback 반환")
            return FALLBACK_ACTION

    def infer_action(self, contents: List[Any]) -> RobotAction:
        """Gemini API 호출 및 예외(503, 429, Timeout) 방어"""
        client = self.get_client()
        config = types.GenerateContentConfig(
            system_instruction=self.system_instruction,
            response_mime_type="application/json",
            response_schema=RobotAction,
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
                return FALLBACK_ACTION

            except APIError as e:
                if "503" in str(e) and attempt < settings.MAX_RETRIES:
                    print(f"[BrainService] 503 과부하 감지. 1초 대기 후 재시도 ({attempt + 1}/{settings.MAX_RETRIES})...")
                    time.sleep(1.0)
                    continue
                elif "429" in str(e):
                    print("[BrainService] 429 Quota Exceeded 감지. 안전 탈출합니다.")
                    return FALLBACK_ACTION
                else:
                    print(f"[BrainService] API 오류 발생: {e}")
                    return FALLBACK_ACTION
            except Exception as e:
                print(f"[BrainService] 예상치 못한 통신 오류: {e}")
                return FALLBACK_ACTION

        return FALLBACK_ACTION