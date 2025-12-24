from sqlalchemy import String, Integer, Text, ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from .db import Base

class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(300), nullable=False)
    version_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # sha256 of file bytes
    file_path: Mapped[str] = mapped_column(String(600), nullable=False)    # local path to the PDF

    chunks: Mapped[list["Chunk"]] = relationship(
        back_populates="document",
        cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("name", "version_hash", name="uq_doc_name_version"),
    )

class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"), nullable=False)

    page_number: Mapped[int] = mapped_column(Integer, nullable=False)     # 1-based
    para_id: Mapped[str] = mapped_column(String(50), nullable=False)      # e.g., "p12-b03"

    # Bounding box on the PDF page (points). Can be used for highlight overlays later.
    x1: Mapped[int] = mapped_column(Integer, nullable=False)
    y1: Mapped[int] = mapped_column(Integer, nullable=False)
    x2: Mapped[int] = mapped_column(Integer, nullable=False)
    y2: Mapped[int] = mapped_column(Integer, nullable=False)

    text: Mapped[str] = mapped_column(Text, nullable=False)

    # 384 dims for all-MiniLM-L6-v2
    embedding: Mapped[list[float]] = mapped_column(Vector(384), nullable=False)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "page_number", "para_id", name="uq_doc_page_para"),
    )
