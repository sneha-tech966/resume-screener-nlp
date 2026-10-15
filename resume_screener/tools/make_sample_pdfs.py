"""
Generates realistic-looking resume PDFs (and a job-description PDF) from the sample TXT files,
so the UI can be demonstrated with 'real' PDF input.   Run:  python tools/make_sample_pdfs.py
Two visual templates are alternated (centered header / left header with coloured headings).
"""
import glob
import os
import re
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer

HEADINGS = ["Summary", "Education", "Skills", "Experience", "Projects"]


def build_pdf(txt_path, out_path, template):
    lines = [l.rstrip() for l in open(txt_path, encoding="utf-8").read().splitlines()]
    name, header_lines, sections, current = lines[0].strip(), [], {}, None
    for l in lines[1:]:
        if l.strip() in HEADINGS:
            current = l.strip()
            sections[current] = []
        elif current is None:
            if l.strip():
                header_lines.append(l.strip())
        elif l.strip():
            sections[current].append(l.strip())

    accent = colors.HexColor("#1f4e79") if template == 0 else colors.HexColor("#7a2e2e")
    align = TA_CENTER if template == 0 else TA_LEFT
    st_name = ParagraphStyle("n", fontName="Helvetica-Bold", fontSize=22, leading=26, alignment=align, textColor=accent)
    st_head = ParagraphStyle("h", fontName="Helvetica", fontSize=9.5, leading=13, alignment=align, textColor=colors.HexColor("#444444"))
    st_sec = ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=11.5, leading=14, spaceBefore=10, spaceAfter=2, textColor=accent)
    st_txt = ParagraphStyle("t", fontName="Helvetica", fontSize=10, leading=13.5)
    st_bold = ParagraphStyle("b", parent=st_txt, fontName="Helvetica-Bold", spaceBefore=3)
    st_bul = ParagraphStyle("u", parent=st_txt, leftIndent=14, bulletIndent=4)

    story = [Paragraph(escape(name.upper() if template == 1 else name), st_name), Spacer(1, 3)]
    for h in header_lines:
        story.append(Paragraph(escape(h), st_head))
    story.append(Spacer(1, 4))
    for sec in HEADINGS:
        if sec not in sections:
            continue
        story.append(Paragraph(sec.upper(), st_sec))
        story.append(HRFlowable(width="100%", thickness=0.8, color=accent, spaceAfter=3))
        for l in sections[sec]:
            if l.startswith("- "):
                story.append(Paragraph(escape(l[2:]), st_bul, bulletText="\u2022"))
            elif sec == "Experience" and re.search(r"(19|20)\d{2}", l):
                story.append(Paragraph(escape(l), st_bold))
            else:
                story.append(Paragraph(escape(l), st_txt))
    SimpleDocTemplate(out_path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
                      topMargin=15 * mm, bottomMargin=15 * mm, title=name, author=name).build(story)


def build_jd_pdf(txt_path, out_path):
    lines = open(txt_path, encoding="utf-8").read().splitlines()
    sty_t = ParagraphStyle("t", fontName="Helvetica-Bold", fontSize=20, leading=24, textColor=colors.HexColor("#1f4e79"))
    sty_h = ParagraphStyle("h", fontName="Helvetica-Bold", fontSize=12, spaceBefore=10, spaceAfter=2)
    sty_p = ParagraphStyle("p", fontName="Helvetica", fontSize=10.5, leading=14)
    sty_b = ParagraphStyle("b", parent=sty_p, leftIndent=14, bulletIndent=4)
    story = [Paragraph(escape(lines[0]), sty_t), Paragraph(escape(lines[1]), sty_p)]
    for l in lines[2:]:
        if not l.strip():
            continue
        if l.startswith("- "):
            story.append(Paragraph(escape(l[2:]), sty_b, bulletText="\u2022"))
        elif l in ("About the Role", "Responsibilities", "Required Skills", "Eligibility"):
            story.append(Paragraph(l, sty_h))
        else:
            story.append(Paragraph(escape(l), sty_p))
    SimpleDocTemplate(out_path, pagesize=A4, title=lines[0]).build(story)


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "data", "resumes_pdf")
    os.makedirs(out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(root, "data", "resumes", "*.txt")),
                   key=lambda f: int(re.findall(r"\d+", f)[-1]))
    for i, f in enumerate(files):
        out = os.path.join(out_dir, os.path.basename(f).replace(".txt", ".pdf"))
        build_pdf(f, out, i % 2)
        print("created", out)
    jd_out = os.path.join(root, "data", "job_description.pdf")
    build_jd_pdf(os.path.join(root, "data", "job_description.txt"), jd_out)
    print("created", jd_out)
