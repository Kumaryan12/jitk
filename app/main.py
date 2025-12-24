from __future__ import annotations
import io
import re
from typing import Any, Optional
from urllib.parse import urlencode
import fitz  # PyMuPDF
from PIL import Image, ImageDraw
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

from .db import engine, Base, SessionLocal
from .models import Chunk, Document
from .retrieval import search_chunks
from .schemas import (
    SuggestRequest,
    SuggestResponse,
    AnswerRequest,
    AnswerResponse,
    AnswerBullet,
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Rendering settings
RENDER_ZOOM = 2.0
HIGHLIGHT_PAD = 6
CROP_PAD = 60  # pixels around bbox when crop=1

app = FastAPI(title="Just-in-Time Knowledge (JITK)")

# ---- CORS (for Next.js on localhost:3000) ----
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model: Optional[SentenceTransformer] = None


@app.on_event("startup")
def on_startup():
    global model
    Base.metadata.create_all(bind=engine)
    model = SentenceTransformer(MODEL_NAME)


@app.get("/")
def health():
    return {"status": "ok"}



# Query builder

def build_query(fields: dict) -> str:
    parts: list[str] = []
    for k, v in fields.items():
        if v is None or v == "":
            continue
        parts.append(f"{k}: {v}")
    return " | ".join(parts) if parts else "general policy guidance"



# Document resolution + rendering

def _resolve_document(db, doc_name: str, doc_version: str | None) -> Document:
    q = db.query(Document).filter(Document.name == doc_name)
    if doc_version:
        doc = q.filter(Document.version_hash == doc_version).first()
    else:
        doc = q.order_by(Document.id.desc()).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


def _render_page_image(pdf_path: str, page_number: int) -> Image.Image:
    pdf = fitz.open(pdf_path)
    try:
        if page_number < 1 or page_number > pdf.page_count:
            raise HTTPException(status_code=404, detail="Invalid page")

        page = pdf.load_page(page_number - 1)
        mat = fitz.Matrix(RENDER_ZOOM, RENDER_ZOOM)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        return Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGBA")
    finally:
        pdf.close()


def _png_response(img: Image.Image) -> Response:
    out = io.BytesIO()
    img.save(out, format="PNG")
    return Response(content=out.getvalue(), media_type="image/png")


def _pdf_bbox_to_pixels(chunk: Chunk) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = chunk.x1, chunk.y1, chunk.x2, chunk.y2
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)

    sx1 = int(x1 * RENDER_ZOOM)
    sy1 = int(y1 * RENDER_ZOOM)
    sx2 = int(x2 * RENDER_ZOOM)
    sy2 = int(y2 * RENDER_ZOOM)
    return sx1, sy1, sx2, sy2



# Provenance helpers

def _hit_to_dict(hit: Any) -> dict:
    if isinstance(hit, dict):
        return hit
    if hasattr(hit, "model_dump"):
        return hit.model_dump()
    if hasattr(hit, "dict"):
        return hit.dict()
    if hasattr(hit, "__dict__"):
        return dict(hit.__dict__)
    raise TypeError(f"Unsupported hit type: {type(hit)}")


def _build_provenance_urls(request: Request, doc_name: str, doc_version: str, page: int, para_id: str) -> dict:
    base = str(request.base_url).rstrip("/")
    page_qs = urlencode({"doc_name": doc_name, "doc_version": doc_version, "page": page})
    hl_qs = urlencode({"doc_name": doc_name, "doc_version": doc_version, "page": page, "para_id": para_id})
    return {
        "page_url": f"{base}/source/page?{page_qs}",
        "highlight_url": f"{base}/source/highlight?{hl_qs}",
    }



# Answer generation (robust)
# Extract -> Cluster -> Compose

_BOILERPLATE_PATTERNS = [
    r"this document is synthetic",
    r"intended only for software demonstrations",
    r"not an insurance contract",
    r"provides no legal guidance",
    r"important notice",
    r"=+",
]


def _split_lines(text: str) -> list[str]:
    raw = re.split(r"[\n\r]+", text or "")
    lines: list[str] = []
    for ln in raw:
        ln = " ".join(ln.strip().split())
        if len(ln) < 30:
            continue
        low = ln.lower()
        if any(re.search(p, low) for p in _BOILERPLATE_PATTERNS):
            continue
        lines.append(ln)
    return lines


def _pick_informative(lines: list[str], k: int = 2) -> list[str]:
    scored: list[tuple[int, str]] = []
    for ln in lines:
        L = len(ln)
        score = 0
        if 60 <= L <= 220:
            score += 3
        elif 30 <= L < 60:
            score += 2
        elif 220 < L <= 320:
            score += 2
        else:
            score += 1
        if ln.isupper():
            score -= 1
        scored.append((score, ln))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [ln for _, ln in scored[:k]]


def _cluster_semantic(pairs: list[tuple[str, AnswerBullet]], st_model: SentenceTransformer, thr: float = 0.82):
    """
    pairs: list of (line_text, bullet)
    returns representative pairs (de-duplicated)
    """
    if not pairs:
        return []

    texts = [t for t, _ in pairs]
    embs = st_model.encode(texts, normalize_embeddings=True)

    used = [False] * len(texts)
    reps: list[tuple[str, AnswerBullet]] = []

    for i in range(len(texts)):
        if used[i]:
            continue
        used[i] = True

        for j in range(i + 1, len(texts)):
            if used[j]:
                continue
            if float(cos_sim(embs[i], embs[j])) >= thr:
                used[j] = True

        reps.append(pairs[i])

    return reps


def _compose_agent_answer(query_text: str, reps: list[tuple[str, AnswerBullet]]) -> str:
    """
    Produces a readable answer *while staying grounded*.
    We only stitch/format retrieved lines; no extra claims.
    """
    if not reps:
        return "No relevant clauses found for this case context."

    # 1) short “agent-facing” summary from top 2 points
    top_summary = []
    for sent, b in reps[:2]:
        top_summary.append(f"{sent} (p.{b.page}, {b.para_id})")
    summary = " ".join(top_summary)

    # 2) cited guidance list
    guidance = []
    for idx, (sent, b) in enumerate(reps, 1):
        guidance.append(
            f"{idx}. {sent} (Source: {b.doc_name} p.{b.page}, {b.para_id})"
        )

    return (
        f"Case context: {query_text}\n\n"
        "Summary (grounded):\n"
        f"{summary}\n\n"
        "Cited guidance:\n" + "\n".join(guidance)
    )



@app.post("/suggest", response_model=SuggestResponse)
def suggest(req: SuggestRequest, request: Request):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    db = SessionLocal()
    try:
        query_text = build_query(req.fields)
        q_emb = model.encode([query_text], normalize_embeddings=True)[0].tolist()

        hits = search_chunks(db, q_emb, top_k=req.top_k)

        enriched: list[dict] = []
        for h in hits:
            d = _hit_to_dict(h)
            dn, dv, pg, pid = d.get("doc_name"), d.get("doc_version"), d.get("page"), d.get("para_id")

            if not (dn and dv and pg and pid):
                raise HTTPException(status_code=500, detail=f"Hit missing provenance keys: {d}")

            d.update(_build_provenance_urls(request, str(dn), str(dv), int(pg), str(pid)))
            enriched.append(d)

        return SuggestResponse(query_used=query_text, suggestions=enriched)
    finally:
        db.close()


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, request: Request):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    db = SessionLocal()
    try:
        query_text = build_query(req.fields)
        q_emb = model.encode([query_text], normalize_embeddings=True)[0].tolist()

        hits = search_chunks(db, q_emb, top_k=req.top_k)

        if not hits:
            return AnswerResponse(
                query_used=query_text,
                answer="No relevant clauses found for this case context.",
                bullets=[],
                sources=[],
            )

        sources: list[dict] = []
        for h in hits:
            d = _hit_to_dict(h)
            dn, dv, pg, pid = d.get("doc_name"), d.get("doc_version"), d.get("page"), d.get("para_id")

            if not (dn and dv and pg and pid):
                raise HTTPException(status_code=500, detail=f"Hit missing provenance keys: {d}")

            d.update(_build_provenance_urls(request, str(dn), str(dv), int(pg), str(pid)))
            sources.append(d)

        
        max_b = min(req.max_bullets, len(sources))
        bullets: list[AnswerBullet] = []
        for i in range(max_b):
            d = sources[i]
            bullets.append(
                AnswerBullet(
                    text=" ".join((d.get("text") or "").split())[:800], 
                    doc_name=str(d["doc_name"]),
                    doc_version=str(d["doc_version"]),
                    page=int(d["page"]),
                    para_id=str(d["para_id"]),
                    page_url=str(d["page_url"]),
                    highlight_url=str(d["highlight_url"]),
                )
            )

       
        candidates: list[tuple[str, AnswerBullet]] = []
        for b in bullets:
            lines = _split_lines(b.text)
            for ln in _pick_informative(lines, k=2):
                candidates.append((ln, b))

       
        reps = _cluster_semantic(candidates, model, thr=0.82)

        
        answer_text = _compose_agent_answer(query_text, reps)

       
        compact_bullets: list[AnswerBullet] = []
        for b in bullets:
            t = " ".join(b.text.split())
            if len(t) > 240:
                t = t[:240].rstrip() + "…"
            compact_bullets.append(
                AnswerBullet(
                    text=t,
                    doc_name=b.doc_name,
                    doc_version=b.doc_version,
                    page=b.page,
                    para_id=b.para_id,
                    page_url=b.page_url,
                    highlight_url=b.highlight_url,
                )
            )

        return AnswerResponse(
            query_used=query_text,
            answer=answer_text,
            bullets=compact_bullets,
            sources=sources,
        )
    finally:
        db.close()


@app.get("/source/page")
def source_page(
    doc_name: str = Query(...),
    doc_version: str | None = Query(None),
    page: int = Query(..., ge=1),
):
    db = SessionLocal()
    try:
        doc = _resolve_document(db, doc_name, doc_version)
        img = _render_page_image(doc.file_path, page)
        return _png_response(img)
    finally:
        db.close()


@app.get("/source/highlight")
def source_highlight(
    doc_name: str = Query(...),
    doc_version: str | None = Query(None),
    page: int = Query(..., ge=1),
    para_id: str = Query(...),
    crop: bool = Query(False),
):
    db = SessionLocal()
    try:
        doc = _resolve_document(db, doc_name, doc_version)

        chunk = (
            db.query(Chunk)
            .filter(
                Chunk.document_id == doc.id,
                Chunk.page_number == page,
                Chunk.para_id == para_id,
            )
            .first()
        )
        if not chunk:
            raise HTTPException(status_code=404, detail="Chunk not found for given doc/page/para_id")

        img = _render_page_image(doc.file_path, page)

        sx1, sy1, sx2, sy2 = _pdf_bbox_to_pixels(chunk)

        sx1 = max(0, sx1 - HIGHLIGHT_PAD)
        sy1 = max(0, sy1 - HIGHLIGHT_PAD)
        sx2 = min(img.width - 1, sx2 + HIGHLIGHT_PAD)
        sy2 = min(img.height - 1, sy2 + HIGHLIGHT_PAD)

        overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        draw.rectangle([sx1, sy1, sx2, sy2], fill=(255, 0, 0, 45))
        draw.rectangle([sx1, sy1, sx2, sy2], outline=(255, 0, 0, 255), width=4)
        img = Image.alpha_composite(img, overlay)

        if crop:
            cx1 = max(0, sx1 - CROP_PAD)
            cy1 = max(0, sy1 - CROP_PAD)
            cx2 = min(img.width - 1, sx2 + CROP_PAD)
            cy2 = min(img.height - 1, sy2 + CROP_PAD)
            img = img.crop((cx1, cy1, cx2, cy2))

        return _png_response(img)
    finally:
        db.close()
