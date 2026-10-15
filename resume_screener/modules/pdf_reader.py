"""
PDF / TXT READER
================
Real resumes are PDFs. This module extracts the text with pypdf and repairs the typical
problems of PDF text before the NLP pipeline sees it:
  * bullet glyphs (bullet, square, private-use Word bullets)  -> "- "
  * ligatures (fi, fl) and curly quotes                          -> plain characters
  * ALL-CAPS headings / names                                    -> Title Case
  * lines that were wrapped by the PDF layout                    -> joined back together
Scanned (image-only) PDFs contain no text; they are reported with a warning (no OCR is used).
"""
import io
import re

from pypdf import PdfReader

from modules import pdf_forensics

_HEADINGS = {
    "summary", "professional summary", "profile", "objective", "career objective", "about me",
    "education", "academic qualifications", "skills", "technical skills", "key skills",
    "experience", "work experience", "professional experience", "internships", "projects",
    "academic projects", "certifications", "achievements", "awards", "languages", "interests",
    "personal projects", "publications", "responsibilities", "required skills", "eligibility",
}
_BULLET_RX = re.compile(r"^[\u2022\u25cf\u25aa\u25a0\u25e6\u2023\u2043\u2219\u00b7\uf0b7\uf0a7\u27a2\u2713\x7f*]+\s*")


def _is_heading(line):
    return line.strip(" :").lower() in _HEADINGS


def clean_pdf_text(raw):
    t = raw
    for a, b in (("\ufb01", "fi"), ("\ufb02", "fl"), ("\u2013", "-"), ("\u2014", "-"), ("\u2018", "'"),
                 ("\u2019", "'"), ("\u201c", '"'), ("\u201d", '"'), ("\xa0", " "), ("\u200b", "")):
        t = t.replace(a, b)
    t = re.sub(r"(\w)-\n([a-z])", r"\1\2", t)                        # hyphenated line breaks
    out, in_body = [], False
    for line in t.splitlines():
        line = line.strip()
        if not line:
            continue
        line = _BULLET_RX.sub("- ", line)
        line = re.sub(r"[\ue000-\uf8ff\x00-\x08\x0b-\x1f\x7f]", "", line).strip()
        if not line:
            continue
        if _is_heading(line):                                         # section heading
            out.append(line.strip(" :").title() if line.isupper() else line.strip(" :"))
            in_body = True
            continue
        if not in_body:                                               # header zone (name, contact)
            if line.isupper() and len(line.split()) <= 4:
                line = line.title()
            out.append(line)
            continue
        prev = out[-1] if out else ""
        wrapped = (not line.startswith("- ") and not _is_heading(prev)
                   and (prev.endswith(",") or line[0].islower()
                        or (len(prev) >= 90 and prev[-1] not in ".!?:;")))
        if wrapped:
            out[-1] = prev + " " + line                               # re-join wrapped line
        else:
            out.append(line)
    return "\n".join(out)


def extract_text(file_bytes, filename):
    """Return (clean_text, info_dict). info has pages, chars, warnings, and the forensic results:
    hidden_spans / hidden_text / hidden_reasons (text a human cannot see - NOT part of the returned text)."""
    info = {"pages": 1, "warnings": [], "hidden_spans": [], "hidden_text": "", "hidden_reasons": {},
            "forensics": "not applicable (TXT file)"}
    name = filename.lower()
    raw = None
    try:
        if name.endswith(".pdf"):
            info["forensics"] = "unavailable (PyMuPDF not installed)"
            if pdf_forensics.available():
                try:
                    res = pdf_forensics.analyze_pdf(file_bytes)
                    raw = res["visible_text"]
                    info["pages"] = res["pages"]
                    info["hidden_spans"] = res["hidden_spans"]
                    info["hidden_text"], info["hidden_reasons"] = pdf_forensics.summarize_hidden(res["hidden_spans"])
                    info["notes"] = res["notes"]
                    info["forensics"] = "checked"
                except Exception as e:                       # fall back to plain text extraction
                    info["forensics"] = f"failed ({e})"
                    raw = None
            if raw is None or len(raw.strip()) < 50:
                reader = PdfReader(io.BytesIO(file_bytes))
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception:
                        info["warnings"].append("PDF is password protected - could not read it.")
                        return "", info
                info["pages"] = len(reader.pages)
                raw = "\n".join((p.extract_text() or "") for p in reader.pages)
        else:
            raw = file_bytes.decode("utf-8", errors="ignore")
    except Exception as e:                                   # corrupt file etc.
        info["warnings"].append(f"Could not read file: {e}")
        return "", info
    text = clean_pdf_text(raw)
    if len(text.strip()) < 50:
        info["warnings"].append("Almost no text found - the PDF may be a scanned image (OCR is not supported).")
    info["chars"] = len(text)
    info["raw_chars"] = len(raw)
    return text, info
