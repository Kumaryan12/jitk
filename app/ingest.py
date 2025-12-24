import hashlib
import fitz  
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from .db import SessionLocal
from .models import Document, Chunk
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def group_blocks(blocks, y_gap: float = 18.0):
    # Keep only non-empty text blocks
    b = []
    for item in blocks:
        x0, y0, x1, y1, txt, *_ = item
        txt = (txt or "").strip()
        if not txt:
            continue
        b.append((float(x0), float(y0), float(x1), float(y1), txt))

    # Sort by vertical position then horizontal
    b.sort(key=lambda t: (t[1], t[0]))

    groups = []
    cur = None  # [x0, y0, x1, y1, text]

    for x0, y0, x1, y1, txt in b:
        if cur is None:
            cur = [x0, y0, x1, y1, txt]
            continue

        prev_y1 = cur[3]
        gap = y0 - prev_y1

        stripped = txt.strip()
        
        letters = re.sub(r"[^A-Za-z]+", "", stripped)
        is_header = (len(letters) >= 10 and letters.isupper() and len(stripped) <= 120)

        starts_section = stripped.startswith(("SECTION", "ENDORSEMENTS", "S1", "S2", "S3", "E9."))

        if gap > y_gap or is_header or starts_section:
            groups.append(tuple(cur))
            cur = [x0, y0, x1, y1, txt]
            cur_text = cur[4].strip()
            cur_letters = re.sub(r"[^A-Za-z]+", "", cur_text)
            cur_is_header = (len(cur_letters) >= 10 and cur_letters.isupper() and len(cur_text) <= 120)

            if cur_is_header:
             groups.append(tuple(cur))
             cur = [x0, y0, x1, y1, txt]
             continue

        else:
            # Merge into current group: union bbox + concat text
            cur[0] = min(cur[0], x0)
            cur[1] = min(cur[1], y0)
            cur[2] = max(cur[2], x1)
            cur[3] = max(cur[3], y1)
            cur[4] = cur[4] + "\n" + txt

    if cur is not None:
        groups.append(tuple(cur))

    return groups


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def embed_texts(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # normalize for cosine similarity
    return model.encode(
        texts,
        normalize_embeddings=True,
        batch_size=32,
        show_progress_bar=True,
    )


def ingest_pdf(pdf_path: str, doc_name: str):
    """
    Ingest a PDF into Postgres:
    - Extract page blocks
    - Group blocks into clause-like chunks (groups)
    - Create embeddings for groups
    - Store chunks with provenance: doc, version, page, para_id, bbox
    """
    model = SentenceTransformer(MODEL_NAME)

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    version_hash = sha256_bytes(pdf_bytes)

    db = SessionLocal()
    try:
        # Upsert document (name + version_hash is identity)
        doc = (
            db.query(Document)
            .filter(Document.name == doc_name, Document.version_hash == version_hash)
            .first()
        )
        if doc is None:
            doc = Document(
                name=doc_name,
                version_hash=version_hash,
                file_path=pdf_path,
            )
            db.add(doc)
            db.commit()
            db.refresh(doc)
        else:
            # Keep file_path in sync
            if getattr(doc, "file_path", None) != pdf_path:
                doc.file_path = pdf_path
                db.commit()

        pdf = fitz.open(stream=pdf_bytes, filetype="pdf")

        chunk_rows: list[dict] = []
        texts: list[str] = []
        total_blocks = 0
        total_groups = 0

        for page_idx in range(pdf.page_count):
            page = pdf.load_page(page_idx)
            page_number = page_idx + 1  # 1-based for humans

            blocks = page.get_text("blocks")
            total_blocks += len(blocks)

            groups = group_blocks(blocks)
            total_groups += len(groups)

            # Optional: debug per page
            # print(f"page {page_number}: blocks={len(blocks)} groups={len(groups)}")

            for g_idx, (x0, y0, x1, y1, text) in enumerate(groups):
                text = (text or "").strip()
                if not text:
                    continue

                para_id = f"p{page_number:03d}-g{g_idx:03d}"  # g = grouped
                texts.append(text)

                chunk_rows.append(
                    {
                        "document_id": doc.id,
                        "page_number": page_number,
                        "para_id": para_id,
                        "x1": int(x0),
                        "y1": int(y0),
                        "x2": int(x1),
                        "y2": int(y1),
                        "text": text,
                    }
                )

        if not texts:
            raise ValueError(
                "No text blocks found. This PDF may be scanned (image-only). "
                "If so, we need OCR."
            )

        embeddings = embed_texts(model, texts)

        inserted = 0
        for row, emb in zip(chunk_rows, embeddings):
            exists = (
                db.query(Chunk)
                .filter(
                    Chunk.document_id == row["document_id"],
                    Chunk.page_number == row["page_number"],
                    Chunk.para_id == row["para_id"],
                )
                .first()
            )
            if exists:
                continue

            db.add(Chunk(**row, embedding=emb.tolist()))
            inserted += 1

        db.commit()

        print(
            f"Ingested: {doc_name} (pages={pdf.page_count}), "
            f"blocks_found={total_blocks}, groups_found={total_groups}, "
            f"inserted={inserted}"
        )

    finally:
        db.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m app.ingest <pdf_path> <doc_name>")
        raise SystemExit(2)

    ingest_pdf(sys.argv[1], sys.argv[2])
