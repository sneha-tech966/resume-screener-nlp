"""
MODULE 2 - INFORMATION EXTRACTION  (Experiment 1 POS tagging + NER practical)
===========================================================================
* spaCy NER           -> PERSON, ORG, GPE/LOC, DATE
* NLTK POS tags       -> nouns / proper nouns as candidate technical terms
* Skill dictionary    -> reliable, case-insensitive, multi-word skill detection
* Small rules         -> university and job-title detection (NER alone is not
                         reliable on short resume lines, so rules back it up)
Missing information is reported as "Not detected" instead of crashing.
"""
import re
import subprocess
import sys
from collections import Counter

import spacy

from modules.preprocessing import SKILL_PATTERNS, SKILLS, pos_tag_text, _stopwords

_nlp = None


def load_spacy():
    """Load the English model (downloads it once if it is missing)."""
    global _nlp
    if _nlp is None:
        try:
            _nlp = spacy.load("en_core_web_sm")
        except OSError:
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            _nlp = spacy.load("en_core_web_sm")
    return _nlp


NOT_DETECTED = "Not detected"
_EDU_WORDS = ("university", "institute", "college", "school", "academy")
_TITLE_WORDS = ("Intern", "Engineer", "Developer", "Analyst", "Scientist", "Administrator",
                "Consultant", "Architect", "Researcher", "Trainee", "Specialist", "Manager")
_EDU_RX = re.compile(
    r"\b(?:[A-Z][A-Za-z.&'-]*[ \t]+){0,5}(?:University|Institute of Technology|College of Engineering|Institute|College)"
    r"(?:[ \t]+of[ \t]+[A-Z][A-Za-z]+(?:[ \t]+[A-Z][A-Za-z]+)*)?")
_T = "|".join(_TITLE_WORDS)
_TITLE_RX = re.compile(r"\b(?:[A-Z][A-Za-z+#]*[ \t]+){0,3}(?:" + _T + r")(?:[ \t]+(?:" + _T + r"))?\b")
# "Machine Learning Intern, DataNest Technologies, Pune, Jun 2024 - Aug 2024"  -> organisation = 2nd comma part
_ORG_RULE_RX = re.compile(r"(?:" + _T + r")[ \t]*,[ \t]*([^,\n]+),")
_COMPANY_SUFFIX = ("technologies", "labs", "lab", "solutions", "systems", "studio", "analytics", "research",
                   "infotech", "software", "ltd", "pvt", "inc", "corp", "services", "ai", "consulting", "group")
_DATE_RX = re.compile(r"(?:[A-Z][a-z]{2}[ \t]+)?(?:19|20)\d{2}[ \t]*-[ \t]*(?:(?:[A-Z][a-z]{2}[ \t]+)?(?:19|20)\d{2}|Present)")
_GENERIC_NOUNS = {
    "experience", "project", "projects", "summary", "education", "skill", "skills", "team", "work",
    "student", "year", "use", "knowledge", "interest", "tasks", "task", "time", "percent", "member",
    "b.tech", "b.e.", "b.sc.", "mba", "b.sc", "b.e", "b.tech.", "intern", "trainee", "graduate",
    "science", "computer", "technology", "information", "engineering", "month", "company",
    "final", "experienced", "enjoys", "passionate", "keen", "building", "enthusiast", "enthusiasts",
    "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec", "percent",
    "good", "solid", "strong", "everyday", "clear", "short", "simple", "weekly", "monthly", "senior",
}
_SECTION_HEADINGS = {"Summary", "Education", "Skills", "Experience", "Projects"}
_HEADER_LOC_RX = re.compile(r"^([A-Z][a-z]+(?:[ ][A-Z][a-z]+)*)(?:,[ ]*([A-Z][A-Za-z ]+))*$")
# every individual word that belongs to some skill ('machine', 'learning', 'scikit', ...)
_SKILL_WORDS = {w for s in SKILLS for w in re.split(r"[^a-z0-9]+", s.lower()) if len(w) > 1}


def _dedupe(items):
    seen, out = set(), []
    for it in items:
        key = it.lower().strip()
        if key and key not in seen:
            seen.add(key)
            out.append(it.strip())
    return out


# ----------------------------------------------------------------------------
def extract_entities(normalized_text):
    """Run spaCy NER on the NORMALIZED text (case preserved) and group entities by label."""
    # NER is run line by line: resume lines are short and independent, and this stops
    # entities from "leaking" across headings and line breaks.
    lines = [l for l in normalized_text.splitlines() if l.strip()]
    ents = {"PERSON": [], "ORG": [], "GPE": [], "LOC": [], "DATE": [], "WORK_OF_ART": []}
    for doc in load_spacy().pipe(lines):
        for e in doc.ents:
            if e.label_ in ents:
                ents[e.label_].append(e.text.strip())
    return {k: _dedupe(v) for k, v in ents.items()}


def extract_skills(normalized_text):
    """Dictionary based skill detection (case-insensitive, multi-word aware).
    Returns canonical skill names in order of first appearance."""
    found = {}
    text = normalized_text
    for rx, canon in SKILL_PATTERNS:
        for m in rx.finditer(text):
            found[canon] = min(found.get(canon, 10 ** 9), m.start())
        text = rx.sub(lambda m: " " * len(m.group(0)), text)   # blank out so shorter skills can't re-match
    return [k for k, _ in sorted(found.items(), key=lambda kv: kv[1])]


def extract_nouns(pos_tags, exclude=(), top_n=10):
    """POS-based extraction: keep NN/NNS/NNP/NNPS tokens, drop stop-words and generic words.
    Words that already belong to a detected skill (python, machine, learning ...) and words that
    are names / places / organisations are excluded, so what remains are domain and job terms."""
    stop = _stopwords()
    counts = Counter()
    for word, tag in pos_tags:
        w = word.strip(".").strip()
        if tag.startswith("NN") and len(w) > 2 and w.lower() not in stop \
                and w.lower() not in _GENERIC_NOUNS and w.lower() not in _SKILL_WORDS \
                and w.lower() not in exclude and not w.isdigit() \
                and not w.lower().endswith("ed") and w.lower() not in {"manages", "enjoys", "builds", "built"}:
            counts[w.lower()] += 1
    return [w for w, _ in counts.most_common(top_n)]


def extract_universities(normalized_text, org_entities):
    unis = [m.group(0).strip() for m in _EDU_RX.finditer(normalized_text)]
    unis += [o.split(",")[-1].strip() for o in org_entities if any(w in o.lower() for w in _EDU_WORDS)]
    unis = [u for u in unis if len(u.split()) >= 2]
    # keep the longest version when one name contains another
    unis = _dedupe(unis)
    return [u for u in unis if not any(u != v and u.lower() in v.lower() for v in unis)]


def extract_job_titles(normalized_text, top_n=3):
    titles = Counter()
    for m in _TITLE_RX.finditer(normalized_text):
        t = m.group(0).strip()
        if t.split()[0].lower() in {"a", "an", "the", "as"}:
            continue
        titles[t] += 1
    return [t for t, _ in titles.most_common(top_n)]


def extract_candidate_name(normalized_text, person_entities):
    """Name = first line if it looks like a name; else a PERSON entity found in the first line;
    else any PERSON entity in the first 3 lines. (Small spaCy models often miss isolated names.)"""
    lines = normalized_text.splitlines()
    first = lines[0].strip() if lines else ""
    if re.fullmatch(r"[A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2}", first):
        return first
    for p in person_entities:
        if p in first and len(p.split()) >= 2:
            return p
    head = " ".join(lines[:3])
    for p in person_entities:
        if p in head and len(p.split()) >= 2:
            return p
    return NOT_DETECTED


def extract_candidate_info(doc_info):
    """Combine all extractors for one preprocessed document (see preprocessing.preprocess_document)."""
    norm = doc_info["normalized"]
    ents = extract_entities(norm)
    name = extract_candidate_name(norm, ents["PERSON"])

    unis = extract_universities(norm, ents["ORG"])
    # Organisations: (a) rule "<Job title>, <Organisation>, ..." and (b) NER ORG candidates that
    # look like companies. NER alone labels degrees / skills as ORG, so it is filtered.
    orgs = [m.group(1).strip() for m in _ORG_RULE_RX.finditer(norm)]
    for o in ents["ORG"]:
        last = o.lower().split()[-1]
        if len(o.split()) >= 2 and last in _COMPANY_SUFFIX and not any(w in o.lower() for w in _EDU_WORDS) and not extract_skills(o):
            orgs.append(o)
    orgs = _dedupe(orgs)

    # Locations: spaCy GPE/LOC + header rule ("Pune, Maharashtra, India" in the first 4 lines).
    header = [l.strip() for l in norm.splitlines()[1:3] if l.strip() not in _SECTION_HEADINGS]
    header_locs = []
    for l in header:
        if _HEADER_LOC_RX.match(l) and not extract_skills(l):
            header_locs += [p.strip() for p in l.split(",")]
    org_words = {w.lower() for w in re.split(r"\W+", " ".join(orgs))}
    ner_locs = [l for l in ents["GPE"] + ents["LOC"]
                if len(l) > 2 and "." not in l and "," not in l
                and not (set(re.split(r"[^a-z0-9]+", l.lower())) & (_SKILL_WORDS | org_words))
                and not extract_skills(l)]
    locs = _dedupe(header_locs + ner_locs)
    # Dates: explicit "Mon YYYY - Mon YYYY" ranges; if none, fall back to NER DATE entities with a year
    dates = _dedupe([m.group(0) for m in _DATE_RX.finditer(norm)])
    if not dates:
        dates = _dedupe([d for d in ents["DATE"] if re.search(r"\b(19|20)\d{2}\b", d)])
    bad = {w.lower() for w in name.split()} | org_words | {w.lower() for l in locs for w in l.split()} \
        | {w.lower() for u in unis for w in re.split(r"\W+", u)}

    return {
        "name": name,
        "universities": unis,
        "organizations": orgs,
        "locations": locs,
        "dates": dates,
        "job_titles": extract_job_titles(norm),
        "skills": extract_skills(norm),
        "nouns": extract_nouns(doc_info["pos_tags"], exclude=bad),
        "entities_raw": ents,
    }


def fmt_list(items, limit=None):
    """Helper for printing: empty list -> 'Not detected'."""
    if not items:
        return NOT_DETECTED
    if limit:
        items = items[:limit]
    return ", ".join(items)
