"""
server/services/embedding_service.py
RAG 장기 기억용 로컬 경량 텍스트 임베딩 서비스.
외부 API 호출 없이 FastEmbed(ONNX 기반, all-MiniLM-L6-v2)로 0.05초 이내 벡터 변환을 수행한다.
"""
import logging
import threading
from typing import Optional

from fastembed import TextEmbedding

from config import settings
from server.schemas.memory import EmbeddingVector

logger = logging.getLogger(__name__)


class EmbeddingService:
    def __init__(self):
        self._model: Optional[TextEmbedding] = None
        self._load_lock = threading.Lock()

    def _get_model(self) -> TextEmbedding:
        """모델 지연 로딩 (최초 1회, Mutex로 중복 로딩 방지)"""
        if self._model is None:
            with self._load_lock:
                if self._model is None:
                    logger.info(f"[EmbeddingService] 로컬 임베딩 모델 로딩: {settings.EMBEDDING_MODEL_NAME}")
                    self._model = TextEmbedding(model_name=settings.EMBEDDING_MODEL_NAME)
        return self._model

    def embed(self, text: str) -> EmbeddingVector:
        """단일 문장을 384차원 임베딩 벡터로 변환한다."""
        model = self._get_model()
        vector = next(iter(model.embed([text])))
        return vector.tolist()


# Spring Bean 스타일 전역 싱글톤 등록
embedding_service = EmbeddingService()
