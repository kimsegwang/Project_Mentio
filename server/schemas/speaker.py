"""
server/schemas/speaker.py
화자 검증(Speaker Verification) 결과 DTO.
"""
from typing import Optional

from pydantic import BaseModel, Field


class SpeakerVerificationResult(BaseModel):
    """
    SpeakerService.verify()의 판정 결과.
    계산된 유사도 점수와 실제 적용된 임계값을 그대로 노출해, AIWorker 등 호출부가
    자체 로그/콘솔 출력에도 실측값을 반영할 수 있도록 한다.
    """

    is_match: bool = Field(description="기준 화자와 동일 인물로 최종 판정되었는지 여부(스킵/소프트패스 포함)")
    similarity: Optional[float] = Field(default=None, description="계산된 코사인 유사도 점수 (검증 자체를 스킵한 경우 None)")
    threshold: Optional[float] = Field(default=None, description="이번 판정에 실제 적용된 임계값 (검증 자체를 스킵한 경우 None)")
    skipped: bool = Field(default=False, description="비활성화 플래그 또는 기준 임베딩 미등록으로 검증 자체를 스킵했는지 여부")
    soft_passed: bool = Field(default=False, description="임계값 미달이지만 세션 소프트패스(최근 통과 이력)로 통과 처리되었는지 여부")
