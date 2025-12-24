from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Chunk, Document


def search_chunks(db: Session, query_embedding: list[float], top_k: int = 8):
    stmt = (
        select(Chunk, Document)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(Chunk.embedding.cosine_distance(query_embedding))
        .limit(top_k)
    )

    results = db.execute(stmt).all()

    hits = []
    for chunk, doc in results:
        hits.append(
            {
                "doc_name": doc.name,
                "doc_version": doc.version_hash,
                "page": chunk.page_number,
                "para_id": chunk.para_id,
                "bbox": {"x1": chunk.x1, "y1": chunk.y1, "x2": chunk.x2, "y2": chunk.y2},
                "text": chunk.text[:800],
            }
        )

    return hits
