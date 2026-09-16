from datetime import datetime
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
