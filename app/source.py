import fitz
from fastapi import HTTPException
from .db import SessionLocal
from .models import Document

def render_page_png(doc_name: str, doc_version: str, page: int) -> bytes:
    db = SessionLocal()
    try:
        doc = db.query(Document).filter_by(name=doc_name, version_hash=doc_version).first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")

        pdf_path = f"docs/{doc_name}.pdf"

        pdf = fitz.open(pdf_path)
        if page < 1 or page > pdf.page_count:
            raise HTTPException(status_code=400, detail="Invalid page number")

        p = pdf.load_page(page - 1)
        pix = p.get_pixmap(dpi=150)
        return pix.tobytes("png")
    finally:
        db.close()
