"""
server/schemas/speaker.py
화자 검증(Speaker Verification) / 화자 식별(Speaker Identification) 결과 DTO.
"""
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

# 화자 임베딩 벡터 타입 별칭 (Resemblyzer d-vector 기준 256차원)
SpeakerEmbeddingVector = List[float]


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


class SpeakerProfile(BaseModel):
    """
    speaker_profiles 테이블 레코드 DTO (임베딩 제외).
    """

    user_id: str = Field(description="화자 고유 식별자 (기존 primary_user 호환)")
    display_name: str = Field(description="사용자에게 노출되는 화자 이름 (예: '아빠', '엄마')")
    is_active: bool = Field(default=True, description="1:N 식별 대상에 포함할지 여부")
    created_at: Optional[datetime] = Field(default=None)
    updated_at: Optional[datetime] = Field(default=None)


class SpeakerIdentificationResult(BaseModel):
    """
    SpeakerService.identify_speaker()의 1:N 판정 결과.
    등록된 모든 활성 화자 임베딩 중 가장 유사도가 높은 화자를 찾아 user_id/display_name으로
    노출한다. 임계값 미달(미등록 화자로 판정)이면 user_id/display_name은 None이다.
    """

    user_id: Optional[str] = Field(default=None, description="최종 판정된 화자의 user_id (미등록/스킵 시 None)")
    display_name: Optional[str] = Field(default=None, description="최종 판정된 화자의 표시 이름 (미등록/스킵 시 None)")
    is_match: bool = Field(description="등록된 화자 중 한 명으로 최종 판정되었는지 여부(스킵/소프트패스 포함)")
    similarity: Optional[float] = Field(default=None, description="가장 가까운 후보와의 코사인 유사도 (검증 자체를 스킵한 경우 None)")
    threshold: Optional[float] = Field(default=None, description="이번 판정에 실제 적용된 임계값 (검증 자체를 스킵한 경우 None)")
    skipped: bool = Field(default=False, description="비활성화 플래그 또는 등록된 화자가 전무하여 식별 자체를 스킵했는지 여부")
    soft_passed: bool = Field(default=False, description="임계값 미달이지만 세션 소프트패스(최근 통과 이력)로 통과 처리되었는지 여부")
