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
from .retrieval import search_chunks, corpus_supports_value, has_whole_word
from .schemas import (
    SuggestRequest,
    SuggestResponse,
    AnswerRequest,
    AnswerResponse,
    AnswerBullet,
)
# add near imports (top of backend/app/main.py)
import re
import time
REQUIRE_STRICT_FIELDS = True
CANDIDATE_POOL_MULT = 8          # retrieve more, then validate evidence
CANDIDATE_POOL_MIN = 40
MIN_RETRIEVAL_SCORE = 0.28  # tune: 0.25–0.35 works well for MiniLM embeddings

_BOILERPLATE_PATTERNS = [
    r"this document is synthetic",
    r"intended only for software demonstrations",
    r"not an insurance contract",
    r"provides no legal guidance",
]

def _is_boilerplate(text: str) -> bool:
    t = (text or "").lower()
    return any(re.search(p, t) for p in _BOILERPLATE_PATTERNS)

def _sentences(text: str) -> list[str]:
    # simple sentence split that works “good enough” for policy clauses
    t = " ".join((text or "").strip().split())
    if not t:
        return []
    # split on sentence-ish boundaries
    parts = re.split(r"(?<=[\.\!\?])\s+", t)
    # keep medium length, filter boilerplate
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 40:
            continue
        if _is_boilerplate(p):
            continue
        out.append(p)
    return out

def _pick_agent_lines(chunks: list[dict], max_lines: int = 3) -> list[tuple[str, dict]]:
    """
    Pick up to max_lines grounded sentences from top retrieved chunks.
    Returns list[(sentence, source_hit_dict)].
    """
    picked: list[tuple[str, dict]] = []
    seen = set()

    for h in chunks:
        for s in _sentences(h.get("text", "")):
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            picked.append((s, h))
            if len(picked) >= max_lines:
                return picked
    return picked

def _compose_agent_answer(lines: list[tuple[str, dict]]) -> str:
    """
    Build a readable answer paragraph. Still grounded: we only use retrieved sentences.
    """
    if not lines:
        return "I don’t know based on the provided case fields — I couldn’t retrieve a strong matching clause."

    # 1–2 sentence summary
    summary_bits = []
    for (s, h) in lines[:2]:
        summary_bits.append(f"{s} (p.{h['page']}, {h['para_id']})")
    summary = " ".join(summary_bits)

    # optional third line as “next step”
    next_step = ""
    if len(lines) >= 3:
        s3, h3 = lines[2]
        next_step = f"\n\nNext relevant clause: {s3} (p.{h3['page']}, {h3['para_id']})"

    return (
        "Agent-ready answer (grounded in retrieved policy clauses):\n"
        f"{summary}"
        f"{next_step}"
    )

import re

_REMOVE_PHRASES = [
    r"^\-+\s*",
    r"\bCite\b",
    r"\bIMPORTANT NOTICE\b",
]

def _clean_fragment(t: str) -> str:
    t = (t or "").strip()
    t = " ".join(t.split())
    for pat in _REMOVE_PHRASES:
        t = re.sub(pat, "", t, flags=re.IGNORECASE).strip()
    # remove duplicated hyphen bullets inside text
    t = re.sub(r"\s*-\s*", " — ", t)
    return t.strip()

def _sentences(text: str) -> list[str]:
    text = _clean_fragment(text)
    # split on sentence boundaries (basic, but works well for PDFs)
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    out = []
    for p in parts:
        p = p.strip()
        if len(p) < 35:
            continue
        # skip common demo boilerplate even if truncated
        low = p.lower()
        if "synthetic" in low or "software demonstration" in low or "not an insurance contract" in low:
            continue
        out.append(p)
    return out

def _keyword_overlap_score(sentence: str, query_text: str) -> int:
    # very simple: boost if sentence contains query tokens
    q = set(re.findall(r"[A-Za-z0-9\-]+", query_text.lower()))
    s = set(re.findall(r"[A-Za-z0-9\-]+", sentence.lower()))
    # ignore tiny tokens
    q = {x for x in q if len(x) >= 4}
    s = {x for x in s if len(x) >= 4}
    return len(q & s)

def _compose_agent_answer_natural(query_text: str, reps: list[tuple[str, "AnswerBullet"]]) -> str:
    """
    reps: list of (representative_sentence_or_fragment, bullet)
    Return: short natural language answer with citations inline.
    """
    if not reps:
        return "No relevant clauses found for this case context."

    # collect candidate sentences with their citations
    cand: list[tuple[int, str, AnswerBullet]] = []
    for frag, b in reps:
        for s in _sentences(frag):
            score = _keyword_overlap_score(s, query_text)
            cand.append((score, s, b))

    # if frag didn’t split well, fallback to cleaned frag
    if not cand:
        for frag, b in reps:
            s = _clean_fragment(frag)
            if len(s) >= 35:
                cand.append((_keyword_overlap_score(s, query_text), s, b))

    # rank best sentences first
    cand.sort(key=lambda x: x[0], reverse=True)

    # pick up to 4 distinct “themes” by avoiding near-duplicates (cheap dedupe)
    chosen: list[tuple[str, AnswerBullet]] = []
    seen = set()
    for _, sent, b in cand:
        key = re.sub(r"[^a-z0-9]+", " ", sent.lower()).strip()
        key = " ".join(key.split()[:10])
        if key in seen:
            continue
        seen.add(key)
        chosen.append((sent, b))
        if len(chosen) >= 4:
            break

    # Build an agent-ready paragraph answer (still grounded)
    lines = []
    lines.append(f"Case context: {query_text}\n")
    lines.append("Decision guidance (grounded + cited):")

    for sent, b in chosen:
        # Ensure it ends like a sentence
        if sent[-1] not in ".!?":
            sent = sent + "."
        lines.append(f"- {sent} (Source: {b.doc_name} p.{b.page}, {b.para_id})")

    lines.append("\nNotes:")
    lines.append("- This answer is constructed only from retrieved policy clauses; open the highlights to verify exact wording.")

    return "\n".join(lines)

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


# --- main.py (replace your /answer endpoint with this) ---

from fastapi import HTTPException, Request
from .schemas import AnswerRequest, AnswerResponse, AnswerBullet
from .retrieval import search_chunks

# Tune these for your demo
MIN_TOP_SCORE = 0.30          # if best hit is below this, say "I don't know"
MIN_GOOD_HITS = 2             # also require at least N decent hits
GOOD_HIT_SCORE = 0.28         # what counts as a "decent" hit


@app.post("/answer", response_model=AnswerResponse)
def answer(req: AnswerRequest, request: Request):
    """
    Strict /answer behavior:
    - Vector retrieval still happens.
    - BUT we only generate an answer if ALL provided fields:
        (a) exist somewhere in the ingested corpus, AND
        (b) are present in the retrieved evidence (top-N sources).
    - Otherwise, return a clear "cannot answer" message + optional citations.
    """

    global model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    db = SessionLocal()
    try:
        fields = req.fields or {}

        # ----------- normalize the inputs we care about -----------
        claim_type = (fields.get("Claim Type") or fields.get("claim_type") or "").strip()
        state = (fields.get("State") or fields.get("state") or "").strip()
        policy_type = (fields.get("Policy Type") or fields.get("policy_type") or "").strip()

        must_match: list[tuple[str, str]] = []
        if claim_type:
            must_match.append(("Claim Type", claim_type))
        if state:
            must_match.append(("State", state))
        if policy_type:
            must_match.append(("Policy Type", policy_type))

        query_text = build_query(fields)
        q_emb = model.encode([query_text], normalize_embeddings=True)[0].tolist()

        # ----------- HARD corpus gating -----------
        # If user value does not exist anywhere in ingested text, do not answer.
        if REQUIRE_STRICT_FIELDS and must_match:
            missing_in_corpus = [(k, v) for (k, v) in must_match if not corpus_supports_value(db, v)]
            if missing_in_corpus:
                msg = ", ".join([f"{k}='{v}'" for k, v in missing_in_corpus])
                return AnswerResponse(
                    query_used=query_text,
                    answer=(
                        f"I can’t generate an answer because the knowledge base contains no mention of: {msg}. "
                        "This usually means the relevant document set was not ingested for that context."
                    ),
                    bullets=[],
                    sources=[],
                )

        # ----------- retrieve larger pool, then validate evidence -----------
        candidate_pool = max(req.top_k * CANDIDATE_POOL_MULT, CANDIDATE_POOL_MIN)
        hits = search_chunks(db, q_emb, top_k=candidate_pool)

        if not hits:
            return AnswerResponse(
                query_used=query_text,
                answer="I couldn't find any relevant clauses for this case context.",
                bullets=[],
                sources=[],
            )

        # ----------- enrich sources with provenance urls -----------
        sources: list[dict] = []
        for h in hits:
            d = _hit_to_dict(h)
            dn, dv, pg, pid = d.get("doc_name"), d.get("doc_version"), d.get("page"), d.get("para_id")
            if not (dn and dv and pg and pid):
                raise HTTPException(status_code=500, detail=f"Hit missing provenance keys: {d}")

            d.update(_build_provenance_urls(request, str(dn), str(dv), int(pg), str(pid)))
            sources.append(d)

        # ----------- Evidence gating: ALL provided fields must appear in retrieved evidence -----------
        # We check top M sources for field matches (collective coverage).
        evidence_window = max(req.top_k, 12)
        evidence_texts = [(s.get("text") or "") for s in sources[:evidence_window]]

        if REQUIRE_STRICT_FIELDS and must_match:
            missing_in_evidence = []
            for label, val in must_match:
                if not any(has_whole_word(val, t) for t in evidence_texts):
                    missing_in_evidence.append((label, val))

            if missing_in_evidence:
                msg = ", ".join([f"{k}='{v}'" for k, v in missing_in_evidence])

                # Return top citations for manual inspection, but NO final answer.
                max_b = min(req.max_bullets, req.top_k, len(sources))
                bullets = []
                for i in range(max_b):
                    d = sources[i]
                    t = " ".join((d.get("text") or "").split())
                    if len(t) > 240:
                        t = t[:240].rstrip() + "…"
                    bullets.append(
                        AnswerBullet(
                            text=t,
                            doc_name=str(d["doc_name"]),
                            doc_version=str(d["doc_version"]),
                            page=int(d["page"]),
                            para_id=str(d["para_id"]),
                            page_url=str(d["page_url"]),
                            highlight_url=str(d["highlight_url"]),
                        )
                    )

                return AnswerResponse(
                    query_used=query_text,
                    answer=(
                        f"I can’t generate a reliable answer because the retrieved evidence does not match all inputs ({msg}). "
                        "This prevents answering the wrong context. You can still inspect the closest clauses below."
                    ),
                    bullets=bullets,
                    sources=sources[: req.top_k],
                )

        # ----------- Confidence gating (after field gating) -----------
        top_score = float(sources[0].get("score") or 0.0)
        num_good = sum(1 for s in sources[:req.top_k] if float(s.get("score") or 0.0) >= GOOD_HIT_SCORE)

        if top_score < MIN_TOP_SCORE or num_good < MIN_GOOD_HITS:
            max_b = min(req.max_bullets, req.top_k, len(sources))
            bullets = []
            for i in range(max_b):
                d = sources[i]
                t = " ".join((d.get("text") or "").split())
                if len(t) > 240:
                    t = t[:240].rstrip() + "…"
                bullets.append(
                    AnswerBullet(
                        text=t,
                        doc_name=str(d["doc_name"]),
                        doc_version=str(d["doc_version"]),
                        page=int(d["page"]),
                        para_id=str(d["para_id"]),
                        page_url=str(d["page_url"]),
                        highlight_url=str(d["highlight_url"]),
                    )
                )

            return AnswerResponse(
                query_used=query_text,
                answer=(
                    "I’m not confident enough to generate an answer from the current retrieval results. "
                    "The closest matches may still be useful—review the cited clauses below."
                ),
                bullets=bullets,
                sources=sources[: req.top_k],
            )

        # ----------- build clean bullets (agent-friendly) -----------
        max_b = min(req.max_bullets, req.top_k, len(sources))
        bullets: list[AnswerBullet] = []
        for i in range(max_b):
            d = sources[i]
            raw = (d.get("text") or "")
            lines = _split_lines(raw)
            picked = _pick_informative(lines, k=2)
            text = " ".join(picked) if picked else " ".join(raw.split())

            text = " ".join(text.split())
            if len(text) > 240:
                text = text[:240].rstrip() + "…"

            bullets.append(
                AnswerBullet(
                    text=text,
                    doc_name=str(d["doc_name"]),
                    doc_version=str(d["doc_version"]),
                    page=int(d["page"]),
                    para_id=str(d["para_id"]),
                    page_url=str(d["page_url"]),
                    highlight_url=str(d["highlight_url"]),
                )
            )

        # ----------- build a readable “agent answer” (still grounded) -----------
        # We create representative lines then de-duplicate.
        candidates: list[tuple[str, AnswerBullet]] = []
        for b in bullets:
            for ln in _pick_informative(_split_lines(b.text), k=1):
                candidates.append((ln, b))

        reps = _cluster_semantic(candidates, model, thr=0.82)

        # This function should NOT hallucinate; it should only reframe reps.
        answer_text = _compose_agent_answer_natural(query_text, reps)

        return AnswerResponse(
            query_used=query_text,
            answer=answer_text,
            bullets=bullets,
            sources=sources[: req.top_k],
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
