from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class BBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class Suggestion(BaseModel):
    doc_name: str
    doc_version: str
    page: int
    para_id: str

    # Optional for non-PDF sources later
    bbox: Optional[BBox] = None

    text: str

    # Added for provenance in UI
    page_url: Optional[str] = None
    highlight_url: Optional[str] = None


class SuggestRequest(BaseModel):
    case_id: str
    user_id: str

    # Allow Appian / UI to send non-string field values too
    fields: Dict[str, Any]

    top_k: int = Field(default=5, ge=1, le=20)


class SuggestResponse(BaseModel):
    query_used: str
    suggestions: List[Suggestion]


# Answer endpoint schemas

class AnswerBullet(BaseModel):
    text: str
    doc_name: str
    doc_version: str
    page: int
    para_id: str
    page_url: str
    highlight_url: str


class AnswerRequest(BaseModel):
    case_id: str
    user_id: str
    fields: Dict[str, Any]
    top_k: int = Field(default=6, ge=1, le=20)
    max_bullets: int = Field(default=4, ge=1, le=10)


class AnswerResponse(BaseModel):
    query_used: str
    answer: str
    bullets: List[AnswerBullet]

    # Keep sources as full Suggestion objects (with URLs + bbox when available)
    sources: List[Suggestion]
