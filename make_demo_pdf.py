from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

INPUT_TXT = "docs/policy_text.txt"
OUTPUT_PDF = "docs/policy.pdf"

def wrap_lines(text, max_len=95):
    words = text.split()
    lines, cur = [], ""
    for w in words:
        if len(cur) + len(w) + 1 <= max_len:
            cur = (cur + " " + w).strip()
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines

def main():
    c = canvas.Canvas(OUTPUT_PDF, pagesize=letter)
    width, height = letter

    with open(INPUT_TXT, "r", encoding="utf-8") as f:
        raw_lines = f.read().splitlines()

    y = height - 50
    for raw in raw_lines:
        # Keep blank lines (creates separate blocks → better bboxes)
        if raw.strip() == "":
            y -= 18
            if y < 60:
                c.showPage()
                y = height - 50
            continue

        # Preserve long lines by wrapping
        lines = raw if len(raw) <= 95 else None
        if lines is None:
            for line in wrap_lines(raw, 95):
                c.drawString(40, y, line)
                y -= 14
                if y < 60:
                    c.showPage()
                    y = height - 50
        else:
            c.drawString(40, y, raw)
            y -= 14
            if y < 60:
                c.showPage()
                y = height - 50

    c.save()
    print(f"Wrote {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
