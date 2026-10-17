from pgvector.sqlalchemy import Vector
from sqlalchemy import Column, Integer, String, Text

from app.core.config import settings
from app.core.db import Base


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String, index=True)
    content = Column(Text)
    # 768 для intfloat/multilingual-e5-base. При смене модели размерность
    # меняется, а таблицу нужно пересоздать (create_all её не изменит).
    embedding = Column(Vector(settings.embedding_dim))
