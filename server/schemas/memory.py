from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

from config import settings

# 임베딩 벡터 타입 별칭 (all-MiniLM-L6-v2 기준 384차원)
EmbeddingVector = List[float]


class MemoryCreateRequest(BaseModel):
    """
    장기 기억 신규 적재 요청 DTO (Background Task에서 사용)
    """
    user_id: str = Field(default=settings.DEFAULT_USER_ID, description="기억 소유 사용자 식별자")
    fact_text: str = Field(min_length=1, description="추출된 대화 사실/에피소드 원문")


class MemoryRecord(BaseModel):
    """
    pgvector 유사도 검색 결과 DTO
    """
    id: int = Field(description="장기 기억 레코드 PK")
    user_id: str = Field(description="기억 소유 사용자 식별자")
    fact_text: str = Field(description="저장된 사실/에피소드 원문")
    similarity: Optional[float] = Field(default=None, description="쿼리 임베딩과의 코사인 유사도 (1에 가까울수록 유사)")
    created_at: Optional[datetime] = Field(default=None, description="기억 적재 시각")


class MemoryRelation(str, Enum):
    """신규 발화와 기존 후보 기억(들) 간의 관계 판정 결과"""
    NEW = "NEW"
    CONTRADICTS = "CONTRADICTS"


class MemoryConflictResult(BaseModel):
    """
    BrainService.classify_memory_relation()의 경량 LLM Reflection 결과 DTO.
    저장 전 근접 중복(Dedup) 판별을 통과했지만 주제가 겹치는 기존 기억이 있을 때,
    사실상 모순(선호/상태 변경)인지 아니면 단순히 무관한 신규 사실인지를 담는다.
    """
    relation: MemoryRelation = Field(description="NEW: 모순 없는 신규 사실 / CONTRADICTS: 기존 기억과 모순")
    conflicting_ids: List[int] = Field(
        default_factory=list,
        description="relation=CONTRADICTS일 때 무효화(is_active=FALSE) 대상 기존 기억의 id 목록",
    )


class UserProfileSummary(BaseModel):
    """
    user_profile_summary 테이블 레코드 DTO.
    MemoryService.summarize_user_profile()이 활성 기억들을 압축해 생성하며,
    대화 컨텍스트 조립 시 "[사용자 프로필: ...]" 형태로 시스템 프롬프트에 주입된다.
    """
    user_id: str = Field(description="기억 소유 사용자 식별자")
    summary_text: str = Field(min_length=1, description="활성 기억을 압축한 고수준 페르소나/선호 성향 요약문")
    source_memory_count: int = Field(default=0, description="요약 생성 시점에 반영된 활성 기억 개수")
    updated_at: Optional[datetime] = Field(default=None, description="요약 갱신 시각")
