from __future__ import annotations
import re
from typing import List, Dict, Any

def _clean_text(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = s.replace("â", "")  # quick fix for mojibake you saw
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s


def _first_useful_line(text: str) -> str:
    
    t = _clean_text(text)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return ""

    # Heuristic: if first line looks like a rule header, keep it
    head = lines[0]
    if re.match(r"^(S\d+|SECTION|\d+(\.\d+)?|E\d+(\.\d+)?)\b", head, flags=re.I):
        return head

    
    return head


def build_answer_from_hits(query_used: str, hits: List[Dict[str, Any]], max_bullets: int = 4):
    
    if not hits:
        return (
            "No relevant policy clauses were retrieved for the given case context.",
            [],
        )

    
    bullets = []
    for h in hits[:max_bullets]:
        title_line = _first_useful_line(h.get("text", ""))
        if not title_line:
            title_line = _clean_text(h.get("text", ""))[:140]

        bullets.append(
            {
                "text": title_line,
                "doc_name": h["doc_name"],
                "doc_version": h["doc_version"],
                "page": int(h["page"]),
                "para_id": str(h["para_id"]),
                "page_url": h["page_url"],
                "highlight_url": h["highlight_url"],
            }
        )

    # A compact answer paragraph: decision + what to cite
    top = hits[0]
    answer = (
        f"Based on the case context ({query_used}), the system retrieved the most relevant policy clauses. "
        f"Start by applying the top-ranked clause from {top['doc_name']} (page {top['page']}, {top['para_id']}). "
        f"Then validate any special conditions/exclusions using the next cited clauses. "
        f"Use the highlighted citations to verify the exact wording for compliance."
    )

    return answer, bullets
