"""
AUTHENTICITY CHECK - keyword stuffing and unsupported skills
============================================================
Two kinds of manipulation are looked for:

 A. HIDDEN TEXT (white / near-white / tiny / off-page text) - found by modules/pdf_forensics.py.
    The hidden text is NOT used for scoring; it is reported here.
 B. VISIBLE KEYWORD STUFFING - a skill is "listed" in the Skills section but never appears in
    Experience / Projects (nothing shows the candidate has used it), the skill list is unusually
    long, or one keyword is repeated many times.

Skill evidence levels:
    demonstrated : appears in Experience / Projects / Certifications / Achievements
    mentioned    : appears only in Summary / Education / header
    listed       : appears only in the Skills section
    unknown      : the resume has no recognisable sections, so it cannot be judged
IMPORTANT: this is an indicator for a human reviewer, not proof of dishonesty. Many honest
resumes list a few skills without a project, so only clear patterns raise a flag.
"""
import re
from collections import Counter

from modules.preprocessing import SKILL_PATTERNS
from modules.extraction import extract_skills

SKILL_SECTIONS = {"skills", "technical skills", "key skills"}
EVIDENCE_SECTIONS = {"experience", "work experience", "professional experience", "internships", "projects",
                     "academic projects", "personal projects", "certifications", "achievements", "awards"}
ALL_HEADINGS = SKILL_SECTIONS | EVIDENCE_SECTIONS | {
    "summary", "professional summary", "profile", "objective", "career objective", "about me", "education",
    "academic qualifications", "languages", "interests", "publications"}

LONG_SKILL_LIST = 22          # claimed distinct skills from which the list is called unusually long
REPEAT_LIMIT = 7              # same skill written this many times
UNSUPPORTED_RATIO_MEDIUM = 0.6
UNSUPPORTED_RATIO_HIGH = 0.8


def split_sections(text):
    """Split resume text into {heading (lower case): text}. Lines before the first heading = 'header'."""
    sections, current = {"header": []}, "header"
    for line in text.splitlines():
        key = line.strip(" :").lower()
        if key in ALL_HEADINGS:
            current = key
            sections.setdefault(current, [])
        else:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


def skill_evidence(text):
    """Return ({skill: level}, can_judge). See module docstring for the levels."""
    sections = split_sections(text)
    claimed = extract_skills(text)
    has_skills = any(k in sections for k in SKILL_SECTIONS)
    has_evidence = any(k in sections and sections[k].strip() for k in EVIDENCE_SECTIONS)
    if not (has_skills and has_evidence):
        return {s: "unknown" for s in claimed}, False
    by_sec = {k: set(extract_skills(v)) for k, v in sections.items()}
    out = {}
    for s in claimed:
        if any(s in by_sec.get(k, ()) for k in EVIDENCE_SECTIONS):
            out[s] = "demonstrated"
        elif any(s in v for k, v in by_sec.items() if k not in SKILL_SECTIONS):
            out[s] = "mentioned"
        else:
            out[s] = "listed"
    return out, True


def keyword_repetition(text):
    counts = Counter()
    for rx, canon in SKILL_PATTERNS:
        n = len(rx.findall(text))
        if n:
            counts[canon] += n
            text = rx.sub(" ", text)
    return {k: v for k, v in counts.items() if v >= REPEAT_LIMIT}


def build_integrity(file_info, visible_text, skills, jd_skills):
    """Full authenticity report for one resume.
    file_info: info dict from pdf_reader (hidden_spans, hidden_text, hidden_reasons ...)
    skills:    skills detected in the visible text
    """
    flags = []
    hidden_text = file_info.get("hidden_text", "")
    hidden_words = len(hidden_text.split())
    hidden_skills = extract_skills(hidden_text) if hidden_text else []
    hidden_jd = [s for s in hidden_skills if s in jd_skills]
    hidden_only_jd = [s for s in hidden_jd if s not in skills]

    if file_info.get("hidden_spans"):
        reasons = ", ".join(f"{k} ({v} words)" for k, v in file_info.get("hidden_reasons", {}).items())
        detail = f"{hidden_words} words that a human reader cannot see: {reasons}."
        if hidden_jd:
            detail += f" They contain {len(hidden_jd)} of the job's required skills: {', '.join(hidden_jd)}."
        flags.append({"severity": "high", "title": "Hidden text found in the PDF", "detail": detail +
                      " This text was ignored when scoring."})

    evidence, can_judge = skill_evidence(visible_text)
    claimed_req = [s for s in jd_skills if s in skills]
    unsupported = [s for s in claimed_req if evidence.get(s) in ("listed", "mentioned")]
    if can_judge and claimed_req:
        ratio = len(unsupported) / len(claimed_req)
        if len(claimed_req) >= 5 and ratio >= UNSUPPORTED_RATIO_HIGH:
            flags.append({"severity": "high", "title": "Required skills claimed without any supporting experience",
                          "detail": f"{len(unsupported)} of {len(claimed_req)} required skills appear only in the skills list "
                                    f"or summary, never in experience or projects: {', '.join(unsupported)}."})
        elif len(claimed_req) >= 5 and ratio >= UNSUPPORTED_RATIO_MEDIUM:
            flags.append({"severity": "medium", "title": "Many required skills are not backed by experience",
                          "detail": f"{len(unsupported)} of {len(claimed_req)} required skills appear only in the skills list "
                                    f"or summary: {', '.join(unsupported)}."})
        elif unsupported:
            flags.append({"severity": "low", "title": "A few skills are listed without supporting experience",
                          "detail": ", ".join(unsupported) + " - worth asking about in an interview."})
    if len(skills) >= LONG_SKILL_LIST:
        flags.append({"severity": "medium", "title": "Unusually long skill list",
                      "detail": f"{len(skills)} different technical skills are listed; most candidates list far fewer."})
    rep = keyword_repetition(visible_text)
    if rep:
        flags.append({"severity": "medium", "title": "Keywords repeated unusually often",
                      "detail": ", ".join(f"{k} ×{v}" for k, v in rep.items())})

    if any(f["severity"] == "high" for f in flags):
        trust = "🚩 High risk"
    elif any(f["severity"] == "medium" for f in flags):
        trust = "⚠️ Review"
    else:
        trust = "✅ OK"

    verified = [s for s in claimed_req if evidence.get(s) in ("demonstrated", "unknown")]
    return {
        "flags": flags, "trust": trust, "evidence": evidence, "can_judge": can_judge,
        "hidden_text": hidden_text, "hidden_words": hidden_words, "hidden_skills": hidden_skills,
        "hidden_jd_skills": hidden_jd, "hidden_only_jd": hidden_only_jd,
        "unsupported_required": unsupported, "verified_required": verified,
        "verified_coverage": 100.0 * len(verified) / len(jd_skills) if jd_skills else 0.0,
        "repetition": rep,
    }
