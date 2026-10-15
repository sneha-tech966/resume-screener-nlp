"""
PDF FORENSICS - detects text that a human reader cannot see
===========================================================
Applicants sometimes add keywords from the job description in white text, microscopic
font or outside the page so that an automated screener reads them but a human does not.

How it is detected (for every text span in the PDF):
  1. TINY FONT      - font size below 4 pt (unreadable for a human).
  2. OFF PAGE       - the text box lies outside the visible page.
  3. CAMOUFLAGED    - the page is RENDERED to an image; the colour of the text is compared with
                      the dominant colour behind it (WCAG contrast ratio). Contrast < 1.5 means
                      the text is practically the same colour as its background (white on white,
                      near-white on white ...). White text on a DARK header band is fine because
                      the background there is dark.
  4. NOT RENDERED   - after rendering, almost no pixel of the span has the text colour, i.e. the
                      text is invisible (render mode 3 / opacity 0) or covered by a shape.

Hidden text is removed from the text that is analysed and reported separately, so it cannot
influence the ranking.
Safety rule: on a SCANNED page (one image covers most of the page) the invisible OCR text layer is
normal, so nothing is flagged there. On ordinary pages hidden text is always flagged, however much there is.
"""
import re

import numpy as np

try:                                   # PyMuPDF; the UI falls back to pypdf if it is missing
    import pymupdf as fitz
except ImportError:                    # older PyMuPDF versions
    try:
        import fitz
    except ImportError:
        fitz = None

ZOOM = 3                               # render scale (3 x 72 dpi = 216 dpi)
MIN_FONT_PT = 4.0
MIN_CONTRAST = 1.5
MIN_INK_FRACTION = 0.003


def available():
    return fitz is not None


def _rgb(c):
    return ((c >> 16) & 255, (c >> 8) & 255, c & 255)


def _luminance(rgb):
    def ch(v):
        v = v / 255.0
        return v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4
    r, g, b = rgb
    return 0.2126 * ch(r) + 0.7152 * ch(g) + 0.0722 * ch(b)


def contrast_ratio(c1, c2):
    """WCAG contrast ratio: 1 = identical colours, 21 = black on white."""
    l1, l2 = _luminance(c1), _luminance(c2)
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def _dominant_colour(region):
    """Most common colour in an image region (quantised) = background behind the text."""
    q = ((region[:, :, 0] >> 4).astype(np.int32) << 8) | ((region[:, :, 1] >> 4).astype(np.int32) << 4) \
        | (region[:, :, 2] >> 4).astype(np.int32)
    flat = q.ravel()
    mode = np.bincount(flat).argmax()
    mask = (q == mode)
    return tuple(int(region[:, :, k][mask].mean()) for k in range(3))


def _classify_span(span, page_rect, img):
    """Return a reason string if the span is not visible to a human, else None."""
    text = span["text"]
    if not text.strip():
        return None
    x0, y0, x1, y1 = span["bbox"]
    if span["size"] < MIN_FONT_PT:
        return f"tiny font ({span['size']:.1f} pt)"
    if x1 < page_rect.x0 or x0 > page_rect.x1 or y1 < page_rect.y0 or y0 > page_rect.y1:
        return "positioned outside the page"
    # pixel region of the span
    px0, py0 = max(int(x0 * ZOOM), 0), max(int(y0 * ZOOM), 0)
    px1, py1 = min(int(x1 * ZOOM) + 1, img.shape[1]), min(int(y1 * ZOOM) + 1, img.shape[0])
    if px1 - px0 < 2 or py1 - py0 < 2:
        return None
    region = img[py0:py1, px0:px1]
    bg = _dominant_colour(region)
    colour = _rgb(span["color"])
    if contrast_ratio(colour, bg) < MIN_CONTRAST:
        return f"text colour matches the background (contrast {contrast_ratio(colour, bg):.2f}:1)"
    dist = np.sqrt(((region.astype(np.int32) - np.array(colour)) ** 2).sum(axis=2))
    if len(text.strip()) >= 3 and (dist < 90).mean() < MIN_INK_FRACTION:
        return "text is not rendered (invisible or covered)"
    return None


def _is_scanned_page(page):
    """True if a single image covers most of the page (scanned document)."""
    area = page.rect.width * page.rect.height
    try:
        for im in page.get_image_info():
            x0, y0, x1, y1 = im["bbox"]
            if (x1 - x0) * (y1 - y0) >= 0.6 * area:
                return True
    except Exception:
        pass
    return False


def analyze_pdf(file_bytes):
    """Returns dict: visible_text, hidden_spans [{text, reason, page, size}], pages, notes."""
    doc = fitz.open(stream=file_bytes, filetype="pdf")
    if doc.needs_pass:
        doc.authenticate("")
    visible_lines, hidden_spans, notes = [], [], []
    for pno, page in enumerate(doc, start=1):
        pix = page.get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM), colorspace=fitz.csRGB, alpha=False)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
        page_vis, page_hid, page_all, total_chars, hid_chars = [], [], [], 0, 0
        for block in page.get_text("dict")["blocks"]:
            if block.get("type") != 0:
                continue
            for line in block["lines"]:
                keep = []
                page_all.append("".join(sp["text"] for sp in line["spans"]))
                for span in line["spans"]:
                    reason = _classify_span(span, page.rect, img)
                    n = len(span["text"].strip())
                    total_chars += n
                    if reason:
                        hid_chars += n
                        page_hid.append({"text": span["text"].strip(), "reason": reason, "page": pno,
                                         "size": round(span["size"], 1)})
                    else:
                        keep.append(span["text"])
                if keep:
                    page_vis.append("".join(keep))
        if total_chars and hid_chars / total_chars > 0.6 and _is_scanned_page(page):
            # a full-page image with an invisible OCR text layer is normal for scanned resumes
            notes.append(f"Page {pno}: scanned page with an OCR text layer - hidden-text check skipped for this page.")
            page_vis, page_hid = page_all, []
        visible_lines.extend(page_vis)
        hidden_spans.extend(page_hid)
    pages = len(doc)
    doc.close()
    return {"visible_text": "\n".join(visible_lines), "hidden_spans": hidden_spans, "pages": pages, "notes": notes}


def summarize_hidden(hidden_spans):
    """Group the hidden spans by reason and build one readable text block."""
    text = " ".join(s["text"] for s in hidden_spans)
    reasons = {}
    for s in hidden_spans:
        key = re.sub(r"\s*\(.*?\)", "", s["reason"])
        reasons[key] = reasons.get(key, 0) + len(s["text"].split())
    return text, reasons
