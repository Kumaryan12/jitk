from __future__ import annotations
import re
from typing import Any, Dict, Optional
from sqlalchemy import func, select
from sqlalchemy.orm import Session
from .models import Chunk, Document

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower()

def has_whole_word(value: str, text: str) -> bool:
    """
    Whole-word match to avoid substring false hits.
    """
    v = re.escape(_norm(value))
    t = _norm(text)
    if not v:
        return True
    return re.search(rf"\b{v}\b", t) is not None

def corpus_supports_value(db: Session, value: str) -> bool:
    """
    True if *any* chunk in the ingested corpus mentions this value.
    Used for strict gating (e.g., State='Goa' should block answers if not present anywhere).
    """
    v = _norm(value)
    if not v:
        return True

    # Fast-ish existence check
    stmt = select(Chunk.id).where(func.lower(Chunk.text).contains(v)).limit(1)
    return db.execute(stmt).first() is not None



def _has_whole_word(value: str, text: str) -> bool:
    """
    True if value appears as a whole word in text (case-insensitive).
    Prevents substring false positives.
    Example: 'go' shouldn't match 'golf'.
    """
    v = re.escape(_norm(value))
    t = _norm(text)
    return re.search(rf"\b{v}\b", t) is not None


def corpus_supports_value(db: Session, value: str) -> bool:
    """
    Checks if the ingested corpus contains this value anywhere in Chunk.text.
    Used for strict validation (e.g., State='Goa' should yield 0 results if not present).
    """
    v = _norm(value)
    if not v:
        return True
    stmt = select(Chunk.id).where(func.lower(Chunk.text).contains(v)).limit(1)
    return db.execute(stmt).first() is not None


def search_chunks(
    db: Session,
    query_embedding: list[float],
    top_k: int = 8,
    fields: Optional[Dict[str, str]] = None,
    candidate_pool: Optional[int] = None,
    min_score: float = 0.20,
    require_state_match: bool = True,
) -> list[dict[str, Any]]:
    """
    Revised retrieval:
    - Pull a larger candidate pool via vector similarity
    - Compute score = 1 - cosine_distance
    - Apply score threshold
    - Apply strict field filtering (State) so wrong fields don't return unrelated clauses
    - Return final top_k hits

    NOTE: This does NOT require DB schema changes.
    """

    fields = fields or {}
    state = (fields.get("State") or fields.get("state") or "").strip()

    pool = candidate_pool or max(top_k * 6, 30)

    distance = Chunk.embedding.cosine_distance(query_embedding).label("distance")
    score = (1 - Chunk.embedding.cosine_distance(query_embedding)).label("score")

    stmt = (
        select(Chunk, Document, distance, score)
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance.asc())
        .limit(pool)
    )

    results = db.execute(stmt).all()

    hits: list[dict[str, Any]] = []
    for chunk, doc, dist, sc in results:
        sc = float(sc) if sc is not None else 0.0
        if sc < min_score:
            continue

        # ✅ Hard State gating: if State is provided, require it to appear in clause text.
        # This prevents inputs like Goa from returning Florida-specific clauses.
        if require_state_match and state:
            if not _has_whole_word(state, chunk.text or ""):
                continue

        hits.append(
            {
                "doc_name": doc.name,
                "doc_version": doc.version_hash,
                "page": int(chunk.page_number),
                "para_id": str(chunk.para_id),
                "bbox": {
                    "x1": int(chunk.x1),
                    "y1": int(chunk.y1),
                    "x2": int(chunk.x2),
                    "y2": int(chunk.y2),
                },
                "text": (chunk.text or "")[:800],
                "distance": float(dist) if dist is not None else None,
                "score": sc,  # higher = better
            }
        )

        if len(hits) >= top_k:
            break

    return hits
