"""
AUTOMATED RESUME SCREENER AND INFORMATION EXTRACTOR USING NLP
=============================================================
Run:  python main.py
      python main.py --query "Python machine learning data analyst"

Pipeline: preprocessing -> information extraction -> representation (BoW / TF-IDF / Word2Vec)
          -> job matching -> skill matching -> IR -> TextRank summary -> clustering -> topics -> report
"""
import argparse
import glob
import importlib.util
import os
import re
import subprocess
import sys
import warnings

warnings.filterwarnings("ignore")


# ==== BOOTSTRAP: install packages / models / NLTK data (needed for Google Colab) ====
def bootstrap():
    pip_names = {"numpy": "numpy", "pandas": "pandas", "nltk": "nltk", "spacy": "spacy",
                 "sklearn": "scikit-learn", "gensim": "gensim", "networkx": "networkx",
                 "matplotlib": "matplotlib", "scipy": "scipy"}
    missing = [pip for mod, pip in pip_names.items() if importlib.util.find_spec(mod) is None]
    env = dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1")
    if missing:
        print("Installing missing packages:", ", ".join(missing))
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing], env=env)
    import spacy
    try:
        spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model en_core_web_sm ...")
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"], env=env)
    import nltk
    wanted = {"punkt": "tokenizers/punkt", "punkt_tab": "tokenizers/punkt_tab",
              "stopwords": "corpora/stopwords", "wordnet": "corpora/wordnet", "omw-1.4": "corpora/omw-1.4",
              "averaged_perceptron_tagger": "taggers/averaged_perceptron_tagger",
              "averaged_perceptron_tagger_eng": "taggers/averaged_perceptron_tagger_eng"}
    for res, path in wanted.items():
        found = False
        for candidate in (path, path + ".zip"):
            try:
                nltk.data.find(candidate)
                found = True
                break
            except LookupError:
                pass
        if not found:                                   # download only what is missing
            print(f"Downloading NLTK resource: {res}")
            nltk.download(res, quiet=True)


bootstrap()
# ==== END BOOTSTRAP ====


# ======================================================================
# MODULE FILE: modules/preprocessing.py
# ======================================================================
"""
MODULE 1 - TEXT PREPROCESSING  (Experiment 1)
=============================================
Two parallel representations are kept for every document:

  1. NORMALIZED TEXT  -> only URLs / e-mails / phones removed, case and
                         punctuation preserved. Used for NER, POS tagging and
                         skill extraction (so "C++", "C#", ".NET" survive).
  2. CLEAN TEXT       -> lowercase, stop-words removed, lemmatized, tech terms
                         protected as single tokens (e.g. "machine_learning").
                         Used for BoW / TF-IDF / Word2Vec / clustering.

Concepts used: lowercasing, URL removal, special character handling,
tokenization, stop-word removal, stemming, lemmatization, POS tagging.
"""
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import sent_tokenize, regexp_tokenize

# ----------------------------------------------------------------------------
# Technical skill dictionary (canonical name -> used for detection everywhere)
# ----------------------------------------------------------------------------
SKILLS = [
    # programming languages
    "Python", "Java", "C", "C++", "C#", "JavaScript", "TypeScript", "SQL", "HTML", "CSS",
    ".NET", "Bash",
    # web / backend
    "React", "Node.js", "Flask", "Django", "Bootstrap", "Spring Boot", "REST API",
    # data / ML / AI
    "Machine Learning", "Deep Learning", "NLP", "Artificial Intelligence", "Data Science",
    "Data Analysis", "Data Visualization", "Computer Vision", "Statistics",
    "Pandas", "NumPy", "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "OpenCV",
    "NLTK", "spaCy", "Hugging Face", "Matplotlib", "CNN", "Jupyter",
    # BI / office
    "Power BI", "Tableau", "Excel",
    # cloud / devops
    "AWS", "Azure", "Docker", "Kubernetes", "Jenkins", "Terraform", "CI/CD", "Linux",
    "Git", "GitHub",
    # databases
    "MongoDB", "MySQL", "PostgreSQL", "Redis",
    # security
    "Network Security", "Penetration Testing", "Cryptography", "Kali Linux", "Wireshark",
    "Firewall",
    # general CS / IT
    "OOP", "Data Structures", "MS Office", "Windows",
]

# alternative spellings -> canonical skill
SKILL_ALIASES = {
    "natural language processing": "NLP",
    "sklearn": "Scikit-learn",
    "scikit learn": "Scikit-learn",
    "postgres": "PostgreSQL",
    "object oriented programming": "OOP",
    "object-oriented programming": "OOP",
    "convolutional neural network": "CNN",
    "convolutional neural networks": "CNN",
    "js": "JavaScript",
}

# the single letter "C" is matched case-sensitively (it must not match a lone letter c)
_CASE_SENSITIVE = {"C"}


def skill_token(skill):
    """Turn a skill into a single safe token: 'C++' -> 'cplusplus', 'Power BI' -> 'power_bi'."""
    t = skill.lower()
    for a, b in (("++", "plusplus"), ("#", "sharp"), (".", "dot"), ("/", ""), ("-", ""), (" ", "_")):
        t = t.replace(a, b)
    return t


def _build_skill_patterns():
    """(regex, canonical) pairs, longest phrases first so 'Machine Learning' wins over 'Learning'."""
    entries = [(s, s) for s in SKILLS] + [(a, c) for a, c in SKILL_ALIASES.items()]
    entries.sort(key=lambda e: -len(e[0]))
    patterns = []
    for phrase, canon in entries:
        flags = 0 if canon in _CASE_SENSITIVE and phrase == canon else re.IGNORECASE
        body = re.escape(phrase).replace(r"\ ", r"[\s\-]+")
        rx = re.compile(r"(?<![A-Za-z0-9+#])" + body + r"(?![A-Za-z0-9+#])", flags)
        patterns.append((rx, canon))
    return patterns


SKILL_PATTERNS = _build_skill_patterns()
TOKEN2SKILL = {skill_token(s): s for s in SKILLS}          # for pretty printing
TOKEN2SKILL["natural_language_processing"] = "NLP"

_stop = None
_stemmer = PorterStemmer()
_lemmatizer = WordNetLemmatizer()


def _stopwords():
    global _stop
    if _stop is None:
        _stop = set(stopwords.words("english"))
    return _stop


# ----------------------------------------------------------------------------
# Step helpers
# ----------------------------------------------------------------------------
def normalize_text(text):
    """NORMALIZED representation: remove URLs, e-mails, phone numbers, bullets; fix whitespace.
    Case and punctuation are kept on purpose (needed by NER / skill detection)."""
    text = re.sub(r"https?://\S+|www\.\S+|\b(?:linkedin|github)\.com\S*", " ", text, flags=re.I)   # URLs
    text = re.sub(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b", " ", text)     # e-mail addresses
    text = re.sub(r"\+?\d[\d\s\-]{8,}\d", " ", text)               # phone numbers
    text = re.sub(r"\b(Email|Phone|GitHub|LinkedIn)\s*:", " ", text, flags=re.I)
    text = re.sub(r"[|\u2022\u25cf\u25aa]", " ", text)             # separators / bullets
    text = re.sub(r"^[\-\*]\s+", "", text, flags=re.M)             # leading '-' or '*'
    text = re.sub(r"[ \t]+", " ", text)                            # normalize spaces
    text = re.sub(r"\n\s*\n+", "\n", text)                         # blank lines
    return text.strip()


_TOKEN_RX = r"\.?[A-Za-z][A-Za-z0-9]*(?:[+#]+|(?:[.\-][A-Za-z][A-Za-z0-9]*)+)*|\d+"


def tokenize_keep_tech(text):
    """Word tokenizer that keeps C++, C#, .NET, Node.js, Scikit-learn as single tokens."""
    return regexp_tokenize(text, _TOKEN_RX)


def split_sentences(normalized):
    """Sentence segmentation. Resumes are line/bullet based, so split on lines first,
    then let NLTK split long lines that contain several sentences."""
    sents = []
    for line in normalized.splitlines():
        line = line.strip()
        if line:
            sents.extend(s.strip() for s in sent_tokenize(line) if s.strip())
    return sents


def protect_skills(text):
    """Replace every known skill by its single-token form ('Machine Learning' -> 'machine_learning')
    so that punctuation removal / tokenization does not destroy it."""
    for rx, canon in SKILL_PATTERNS:
        text = rx.sub(" " + skill_token(canon) + " ", text)
    return text


def clean_text(text):
    """CLEAN representation for vector models. Returns (tokens_no_stop, stems, lemmas)."""
    t = protect_skills(normalize_text(text))
    t = t.lower()                                        # lowercase conversion
    t = re.sub(r"[^a-z0-9_\s]", " ", t)                  # special character removal
    t = re.sub(r"\b\d+\b", " ", t)                       # stand-alone numbers add no meaning
    tokens = t.split()                                   # tokenization
    stop = _stopwords()
    tokens = [w for w in tokens if w not in stop and len(w) > 1]   # stop-word removal
    stems = [_stemmer.stem(w) for w in tokens]           # stemming (crude, for demonstration)
    lemmas = [w if w in TOKEN2SKILL else _lemmatizer.lemmatize(w) for w in tokens]  # lemmatization (skills stay intact: pandas != panda)
    return tokens, stems, lemmas


def clean_for_tfidf(text):
    """Convenience: full cleaning -> one string of lemmas (used for queries / job descriptions)."""
    return " ".join(clean_text(text)[2])


def pos_tag_text(normalized):
    """POS tagging (Experiment 1) on the NORMALIZED text so case information helps the tagger."""
    tokens = tokenize_keep_tech(normalized)
    return nltk.pos_tag(tokens)


# ----------------------------------------------------------------------------
# Full pipeline for one document
# ----------------------------------------------------------------------------
def preprocess_document(raw_text):
    """Run the whole preprocessing pipeline and return every intermediate result in a dict."""
    normalized = normalize_text(raw_text)
    tokens, stems, lemmas = clean_text(raw_text)
    sentences = split_sentences(normalized)
    return {
        "raw": raw_text,
        "normalized": normalized,
        "tokens": tokens,
        "stems": stems,
        "lemmas": lemmas,
        "clean_text": " ".join(lemmas),
        "pos_tags": pos_tag_text(normalized),
        "sentences": sentences,
        "sentence_tokens": [clean_text(s)[2] for s in sentences],   # for Word2Vec
    }


def pretty_token(tok):
    """Show 'scikitlearn' as 'Scikit-learn', 'machine_learning' as 'Machine Learning' etc."""
    if tok in TOKEN2SKILL:
        return TOKEN2SKILL[tok]
    return tok.replace("_", " ")


# ======================================================================
# MODULE FILE: modules/extraction.py
# ======================================================================
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


# ======================================================================
# MODULE FILE: modules/representation.py
# ======================================================================
"""
MODULE 3 - TEXT REPRESENTATION  (Experiment 2)
==============================================
Three ways of turning text into numbers:

  A. Bag of Words  - raw word counts. Simple baseline, every word counts the same.
  B. TF-IDF        - counts re-weighted so words that are frequent in ONE resume but rare in
                     the others get a high weight. This is our main representation because
                     it highlights what is distinctive about a candidate.
  C. Word2Vec      - dense vectors learned from word context; words used in similar contexts
                     (e.g. 'pandas' and 'numpy') get similar vectors. TF-IDF cannot do this.
Inputs are the CLEAN (lemmatized, skill-protected) texts from preprocessing.
"""
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



def build_bow(clean_docs):
    """Bag of Words / Document-Term Matrix."""
    vec = CountVectorizer()
    matrix = vec.fit_transform(clean_docs)
    return vec, matrix


def bow_preview(vec, matrix, n_terms=6, n_docs=3):
    """Small DataFrame preview of the document-term matrix (never the full matrix)."""
    terms = vec.get_feature_names_out()[:n_terms]
    df = pd.DataFrame(matrix[:n_docs, :n_terms].toarray(), columns=[pretty_token(t) for t in terms],
                      index=[f"Resume {i + 1}" for i in range(n_docs)])
    return df


def build_tfidf(clean_docs):
    """TF-IDF fitted on the resumes only (the job description is treated as an unseen QUERY,
    exactly like a search query in Information Retrieval)."""
    vec = TfidfVectorizer(sublinear_tf=True)
    matrix = vec.fit_transform(clean_docs)
    return vec, matrix


def top_tfidf_terms(vec, matrix, doc_index, n=8):
    """Most informative terms of one resume = highest TF-IDF weights."""
    terms = vec.get_feature_names_out()
    row = matrix[doc_index].toarray().ravel()
    idx = row.argsort()[::-1][:n]
    return [(pretty_token(terms[i]), float(row[i])) for i in idx if row[i] > 0]


def train_word2vec(sentence_tokens, vector_size=50, epochs=200):
    """Train a small skip-gram Word2Vec on resume sentences. With only a handful of resumes the
    model is tiny, so similarities are indicative only (it is a demonstration of the concept)."""
    model = Word2Vec(sentences=sentence_tokens, vector_size=vector_size, window=5, min_count=1,
                     sg=1, epochs=epochs, seed=42, workers=1)
    return model


def word_similarity(model, w1, w2):
    """Cosine similarity of two words in Word2Vec space (None if a word is unknown)."""
    if w1 in model.wv and w2 in model.wv:
        return float(model.wv.similarity(w1, w2))
    return None


def document_vector(model, tokens):
    """Average of the word vectors of a document = simple document embedding."""
    vecs = [model.wv[t] for t in tokens if t in model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size)


# ======================================================================
# MODULE FILE: modules/matching.py
# ======================================================================
"""
MODULES 4, 5, 6 - JOB MATCHING, SKILL MATCHING, INFORMATION RETRIEVAL
=====================================================================
* Cosine similarity between TF-IDF vectors  (resume vs job description)
* Rule-based skill coverage  (matched / missing / additional skills)
* IR pipeline: query -> preprocess -> TF-IDF vector -> cosine -> ranking
NOTE: scores measure TEXTUAL SIMILARITY / SKILL COVERAGE only. They are not hiring accuracy.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity



def query_vector(vec, text):
    """Apply the SAME preprocessing + SAME fitted vectorizer to a query / job description."""
    return vec.transform([clean_for_tfidf(text)])


def cosine_scores(query_vec, doc_matrix):
    """cos(q, d) = q.d / (|q||d|) for every document. Returns a 1-D numpy array."""
    return cosine_similarity(query_vec, doc_matrix).ravel()


def rank(scores, labels):
    """Sort labels by score (high -> low). Returns list of (label, score, rank)."""
    order = np.argsort(-scores, kind="stable")
    return [(labels[i], float(scores[i]), r + 1) for r, i in enumerate(order)]


def skill_match(jd_skills, resume_skills):
    """Matched / missing / additional skills + coverage % (matched required / total required)."""
    jd_set, res_set = set(jd_skills), set(resume_skills)
    matched = [s for s in jd_skills if s in res_set]
    missing = [s for s in jd_skills if s not in res_set]
    additional = [s for s in resume_skills if s not in jd_set]
    pct = 100.0 * len(matched) / len(jd_skills) if jd_skills else 0.0
    return {"matched": matched, "missing": missing, "additional": additional, "percentage": pct}


def retrieve(query_text, vec, doc_matrix, labels, top_k=5):
    """Information Retrieval: rank resumes (documents) for a free-text query."""
    q = query_vector(vec, query_text)
    scores = cosine_scores(q, doc_matrix)
    return rank(scores, labels)[:top_k]


# ======================================================================
# MODULE FILE: modules/summarization.py
# ======================================================================
"""
MODULE 7 - TEXTRANK EXTRACTIVE SUMMARY
======================================
Sentences are nodes of a graph; edge weight = cosine similarity of their TF-IDF vectors.
PageRank then scores each sentence: a sentence similar to many other sentences is central,
so it is a good summary candidate. The top sentences are shown in original order.
"""
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity



def _candidate_sentences(sentences):
    """Keep real sentences: >= 6 words, not a bare comma-separated skill list, not a heading,
    and not the header/contact line (those contain no verbs worth summarizing)."""
    keep = []
    for s in sentences:
        words = s.split()
        if len(words) < 6:
            continue
        if s.count(",") >= 4 and not s.rstrip().endswith("."):
            continue
        if "20" in s and " - " in s and s.count(",") >= 2:    # 'Intern, Company, City, Jun 2024 - Aug 2024'
            continue
        keep.append(s)
    return keep


def textrank_summary(sentences, n_sentences=3):
    cands = _candidate_sentences(sentences)
    if len(cands) <= n_sentences:
        return " ".join(cands) if cands else "Not enough text to summarize."
    try:
        cleaned = [clean_for_tfidf(s) or "empty" for s in cands]
        tfidf = TfidfVectorizer().fit_transform(cleaned)           # TF-IDF sentence vectors
        sim = cosine_similarity(tfidf)                              # sentence-sentence similarity
        np.fill_diagonal(sim, 0)                                    # no self loops
        graph = nx.from_numpy_array(sim)                            # weighted sentence graph
        scores = nx.pagerank(graph, weight="weight")                # TextRank = PageRank on this graph
    except Exception:
        return " ".join(cands[:n_sentences])                        # safe fallback
    top = sorted(sorted(scores, key=scores.get, reverse=True)[:n_sentences])   # restore original order
    return " ".join(cands[i].rstrip(".") + "." for i in top)


# ======================================================================
# MODULE FILE: modules/clustering.py
# ======================================================================
"""
MODULE 8 - RESUME CLUSTERING
============================
* Hierarchical (agglomerative) clustering with COSINE distance + average linkage, drawn as a
  dendrogram. The number of clusters is chosen with the silhouette score (an internal metric).
* DBSCAN (density based): finds dense groups and marks isolated resumes as noise (-1).
Cluster labels are derived from the highest TF-IDF terms of each cluster's centroid.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # draw to a file, no window needed
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import pdist
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score



def hierarchical_clustering(tfidf_matrix, labels, out_path="dendrogram.png", k=None, max_k=5):
    X = tfidf_matrix.toarray()
    dist = pdist(X, metric="cosine")
    Z = linkage(dist, method="average")                      # cosine distance + average linkage
    n = X.shape[0]

    best_k, best_s, scores = k, None, {}
    if k is None:
        for kk in range(2, min(max_k, n - 1) + 1):
            lab = fcluster(Z, t=kk, criterion="maxclust")
            if len(set(lab)) > 1:
                scores[kk] = silhouette_score(X, lab, metric="cosine")
        if scores:
            best_k = max(scores, key=scores.get)
            best_s = scores[best_k]
        else:
            best_k = 1
    assignment = fcluster(Z, t=best_k, criterion="maxclust") - 1    # start at 0

    # dendrogram image
    fig, ax = plt.subplots(figsize=(10, 5))
    dendrogram(Z, labels=labels, leaf_rotation=45, leaf_font_size=9, ax=ax)
    ax.set_title("Hierarchical Clustering of Resumes (cosine distance, average linkage)")
    ax.set_ylabel("Cosine distance")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return assignment, best_k, best_s, scores


def dbscan_clustering(tfidf_matrix, eps=0.88, min_samples=2):
    """DBSCAN on cosine distance. eps = max distance for two resumes to be neighbours.
    eps was chosen by trying 0.75-0.92: below ~0.8 everything is noise, above ~0.9 everything merges."""
    model = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine")
    return model.fit_predict(tfidf_matrix.toarray()).astype(int)


def cluster_keywords(tfidf_matrix, vec, assignment, n_terms=4):
    """Label each cluster with the top TF-IDF terms of its centroid."""
    terms = vec.get_feature_names_out()
    X = tfidf_matrix.toarray()
    out = {}
    for c in sorted(set(assignment)):
        if c == -1:
            out[c] = "noise / outliers"
            continue
        centroid = X[assignment == c].mean(axis=0)
        out[c] = ", ".join(pretty_token(terms[i]) for i in centroid.argsort()[::-1][:n_terms])
    return out


# ======================================================================
# MODULE FILE: modules/topic_modeling.py
# ======================================================================
"""
MODULE 9 - TOPIC MODELLING (NMF)
================================
NMF factorizes the TF-IDF matrix V (resumes x words) into W (resumes x topics) and
H (topics x words). Each row of H is a topic = a group of words that appear together.
NMF is used instead of LDA because it is stable and works well on very small corpora.
"""
import numpy as np
from sklearn.decomposition import NMF



def nmf_topics(tfidf_matrix, vec, n_topics=4, n_words=8):
    n_topics = max(1, min(n_topics, tfidf_matrix.shape[0] - 1))
    model = NMF(n_components=n_topics, init="nndsvd", random_state=42, max_iter=500)
    W = model.fit_transform(tfidf_matrix)
    terms = vec.get_feature_names_out()
    topics = []
    for comp in model.components_:
        top = comp.argsort()[::-1][:n_words]
        topics.append([pretty_token(terms[i]) for i in top])
    dominant = W.argmax(axis=1)                 # most important topic of every resume
    return topics, dominant


# ======================================================================
# EMBEDDED SAMPLE DATA
# ======================================================================
SAMPLE_RESUMES = {'resume_1.txt': 'Aarav Mehta\nEmail: aarav.mehta@example.com | Phone: +91 98765 43210 | GitHub: https://github.com/aaravmehta\nPune, Maharashtra, India\n\nSummary\nFinal year Computer Science student with a strong interest in machine learning and data science. Experienced in building predictive models with Python, Pandas and Scikit-learn. Enjoys turning messy data into clear insights.\n\nEducation\nB.Tech in Computer Science, Savitribai Phule Pune University, 2021 - 2025\n\nSkills\nPython, Machine Learning, Data Analysis, Pandas, NumPy, Scikit-learn, SQL, Matplotlib, Statistics, Git, Jupyter\n\nExperience\nMachine Learning Intern, DataNest Technologies, Pune, Jun 2024 - Aug 2024\n- Cleaned and analysed customer datasets using Pandas and NumPy.\n- Built a churn prediction model with Scikit-learn and improved recall by 12 percent.\n- Wrote SQL queries to extract training data from a PostgreSQL warehouse.\n- Used Git for version control and documented experiments for the team.\n\nProjects\n- House Price Prediction: trained regression models in Python and compared them using cross validation.\n- Customer Segmentation: applied machine learning clustering techniques on retail data and visualised results with Matplotlib.\n', 'resume_2.txt': 'Sneha Kulkarni\nsneha.k@example.com | +91 91234 56780 | https://www.linkedin.com/in/snehakulkarni\nMumbai, India\n\nSummary\nFront-end focused web developer who builds responsive and accessible websites. Comfortable with the full JavaScript stack and passionate about clean user interfaces.\n\nEducation\nB.E. in Information Technology, University of Mumbai, 2020 - 2024\n\nSkills\nJavaScript, TypeScript, React, Node.js, HTML, CSS, Bootstrap, MongoDB, Git, GitHub\n\nExperience\nWeb Developer Intern, PixelCraft Studio, Mumbai, Jan 2024 - Jun 2024\n- Developed reusable React components for an e-commerce dashboard.\n- Designed responsive layouts with HTML, CSS and Bootstrap for mobile and desktop.\n- Built REST endpoints using Node.js and stored data in MongoDB.\n- Collaborated with designers through GitHub pull requests and Git branching.\n\nProjects\n- Online Food Ordering Website: a React and Node.js application with user login and cart features.\n- Portfolio Website: a fast responsive site deployed with GitHub pages.\n', 'resume_3.txt': 'Rohan Deshmukh\nrohan.d@example.com | +91 99887 76655 | https://github.com/rohand-sec\nAurangabad, Maharashtra\n\nSummary\nCybersecurity enthusiast focused on network defence, vulnerability assessment and ethical hacking. Participates regularly in capture the flag competitions.\n\nEducation\nB.Tech in Computer Engineering, Dr. Babasaheb Ambedkar Marathwada University, 2021 - 2025\n\nSkills\nNetwork Security, Penetration Testing, Cryptography, Kali Linux, Wireshark, Linux, Firewall, Python, Git\n\nExperience\nSecurity Analyst Intern, CyberShield Labs, Pune, May 2024 - Jul 2024\n- Performed vulnerability scans and penetration testing on internal web applications.\n- Monitored network traffic with Wireshark to detect suspicious activity and possible attacks.\n- Wrote Python scripts to automate log analysis and report security incidents.\n- Documented firewall rules and recommended security improvements.\n\nProjects\n- Intrusion Detection Lab: configured a firewall and monitored attacks in a virtual network.\n- Password Cracking Study: compared cryptography techniques and password hashing methods.\n', 'resume_4.txt': 'Priya Nair\npriya.nair@example.com | +91 90000 11122 | https://www.linkedin.com/in/priyanair\nBengaluru, Karnataka\n\nSummary\nData analytics graduate who enjoys building dashboards and explaining business trends with data. Skilled in SQL, Excel and visual storytelling.\n\nEducation\nB.Sc. in Statistics, Christ University, 2020 - 2023\nMBA in Business Analytics, Symbiosis Institute, 2023 - 2025\n\nSkills\nSQL, Python, Pandas, Excel, Power BI, Tableau, Data Analysis, Statistics, Data Visualization\n\nExperience\nData Analyst Intern, BrightPath Analytics, Bengaluru, Jun 2023 - Sep 2023\n- Built interactive Power BI and Tableau dashboards for the sales team.\n- Performed data analysis on sales records using SQL and Pandas.\n- Automated monthly reports in Excel and reduced manual work by 30 percent.\n- Presented weekly business insights to senior managers.\n\nProjects\n- Retail Sales Dashboard: created a Power BI report showing revenue trends and regional performance.\n- Customer Survey Analysis: used statistics and Python to summarise survey responses.\n', 'resume_5.txt': 'Kabir Shaikh\nkabir.shaikh@example.com | +91 98220 33445 | https://github.com/kabirshaikh\nNagpur, Maharashtra\n\nSummary\nSoftware developer with solid knowledge of object oriented programming and data structures. Enjoys building reliable backend applications in Java and C++.\n\nEducation\nB.E. in Computer Science, Rashtrasant Tukadoji Maharaj Nagpur University, 2020 - 2024\n\nSkills\nJava, C++, C#, .NET, Spring Boot, OOP, Data Structures, MySQL, SQL, Git\n\nExperience\nSoftware Developer Intern, Orion Software Pvt Ltd, Nagpur, Jan 2024 - Jun 2024\n- Developed backend services using Java and Spring Boot for an inventory system.\n- Wrote C# and .NET modules for an internal billing tool.\n- Optimised MySQL queries and designed database tables.\n- Followed agile practices and managed code with Git.\n\nProjects\n- Library Management System: a desktop application in Java with MySQL storage.\n- Chess Engine: implemented game logic and search algorithms in C++.\n', 'resume_6.txt': 'Ananya Iyer\nananya.iyer@example.com | +91 97000 22334 | https://github.com/ananyaiyer\nChennai, Tamil Nadu\n\nSummary\nAI and natural language processing student passionate about language models and text analytics. Has hands-on experience with NLP pipelines, deep learning and machine learning in Python.\n\nEducation\nB.Tech in Artificial Intelligence and Data Science, Anna University, 2021 - 2025\n\nSkills\nPython, NLP, Natural Language Processing, Machine Learning, Deep Learning, PyTorch, Scikit-learn, NumPy, NLTK, spaCy, Hugging Face, Git\n\nExperience\nNLP Research Intern, LinguaTech Research, Chennai, May 2024 - Aug 2024\n- Built text classification models using Scikit-learn, NLTK and spaCy.\n- Fine-tuned transformer models for sentiment analysis with PyTorch and Hugging Face.\n- Implemented named entity recognition and topic modelling experiments in Python.\n- Maintained experiment code on Git and wrote short technical reports.\n\nProjects\n- Fake News Detector: applied NLP techniques and machine learning classifiers to news articles.\n- Question Answering Chatbot: developed a retrieval based chatbot using TF-IDF and cosine similarity.\n', 'resume_7.txt': 'Vikram Patil\nvikram.patil@example.com | +91 98500 44556 | https://www.linkedin.com/in/vikrampatil\nHyderabad, Telangana\n\nSummary\nCloud and DevOps engineer who automates deployments and manages scalable infrastructure. Focused on reliability, monitoring and continuous delivery.\n\nEducation\nB.Tech in Information Technology, Osmania University, 2020 - 2024\n\nSkills\nAWS, Azure, Docker, Kubernetes, Jenkins, Terraform, CI/CD, Linux, Git, GitHub, Python\n\nExperience\nDevOps Engineer Intern, CloudNimbus Solutions, Hyderabad, Feb 2024 - Jul 2024\n- Containerised microservices with Docker and deployed them on Kubernetes.\n- Created CI/CD pipelines using Jenkins and GitHub for automatic testing and release.\n- Provisioned AWS and Azure infrastructure with Terraform scripts.\n- Wrote Python scripts to monitor servers and automate Linux maintenance tasks.\n\nProjects\n- Auto Scaling Web App: deployed a web application on AWS with load balancing and monitoring.\n- Pipeline Templates: built reusable Jenkins pipeline templates for the engineering team.\n', 'resume_8.txt': 'Meera Joshi\nmeera.joshi@example.com | +91 97654 11223 | https://github.com/meerajoshi\nPune, Maharashtra\n\nSummary\nComputer vision enthusiast interested in image recognition and deep learning. Has built convolutional neural networks for classification and object detection.\n\nEducation\nB.E. in Electronics and Telecommunication, College of Engineering Pune, 2021 - 2025\n\nSkills\nComputer Vision, Deep Learning, Machine Learning, Python, OpenCV, TensorFlow, Keras, NumPy, CNN, Git\n\nExperience\nComputer Vision Intern, VisionWorks AI, Pune, Jun 2024 - Sep 2024\n- Trained convolutional neural networks with TensorFlow and Keras to classify product images.\n- Used OpenCV for image preprocessing, augmentation and object detection.\n- Improved model accuracy through hyperparameter tuning and data augmentation.\n- Managed project code and model versions using Git.\n\nProjects\n- Traffic Sign Recognition: a CNN built in Python that classifies road signs from camera images.\n- Face Mask Detection: real-time detection system using OpenCV and deep learning.\n', 'resume_9.txt': 'Arjun Reddy\narjun.reddy@example.com | +91 98490 55667 | https://github.com/arjunreddy\nHyderabad, Telangana\n\nSummary\nBackend and database developer who designs efficient schemas and reliable APIs. Interested in query optimisation and scalable data storage.\n\nEducation\nB.Tech in Computer Science, Jawaharlal Nehru Technological University, 2020 - 2024\n\nSkills\nSQL, MySQL, PostgreSQL, MongoDB, Redis, Node.js, Python, Flask, REST API, Docker, Git\n\nExperience\nBackend Developer Intern, DataBridge Systems, Hyderabad, Jan 2024 - Jun 2024\n- Designed relational database schemas in PostgreSQL and MySQL for a booking platform.\n- Optimised slow SQL queries and added indexes to improve response time.\n- Built REST API services with Node.js and Flask.\n- Cached frequent requests in Redis and deployed services with Docker.\n\nProjects\n- Hotel Booking API: a Flask and PostgreSQL service with authentication and payment records.\n- Inventory Database: designed normalised MySQL tables and stored procedures.\n', 'resume_10.txt': 'Neha Chavan\nneha.chavan@example.com | +91 90110 99887\nAurangabad, Maharashtra\n\nSummary\nIT support graduate with good communication skills and experience helping users solve everyday computer problems. Keen to grow in the technology industry.\n\nEducation\nB.Sc. in Information Technology, Government College of Arts and Science, 2021 - 2024\n\nSkills\nMS Office, Excel, Windows, Troubleshooting, Networking basics, Customer support, Basic SQL\n\nExperience\nIT Support Trainee, Sunrise Infotech, Aurangabad, Jul 2024 - Dec 2024\n- Resolved hardware and software issues for office staff and maintained a helpdesk log.\n- Installed Windows updates and configured printers and basic network connections.\n- Prepared simple Excel sheets to track inventory of computer equipment.\n- Trained new employees to use email and office applications.\n\nProjects\n- Helpdesk Ticket Tracker: created an Excel based tracker for user complaints.\n'}
SAMPLE_JD = 'Machine Learning Intern\nNimbus AI Labs, Pune, India\n\nAbout the Role\nNimbus AI Labs is looking for a Machine Learning Intern to join our applied data team for a six month internship. You will work with senior engineers to clean data, build predictive models and present findings to the product team.\n\nResponsibilities\n- Collect, clean and explore datasets using Python, Pandas and NumPy.\n- Perform data analysis to find patterns and prepare features for machine learning models.\n- Build and evaluate machine learning models using Scikit-learn.\n- Write SQL queries to extract data from relational databases.\n- Work on simple NLP tasks such as text classification and information extraction.\n- Use Git for version control and collaborate through code reviews.\n\nRequired Skills\nPython, Machine Learning, Data Analysis, Pandas, NumPy, Scikit-learn, SQL, Natural Language Processing (NLP), Git.\n\nEligibility\nFinal year students or recent graduates in Computer Science, Information Technology or a related field, with good communication skills and a strong interest in learning.\n'


def ensure_sample_data(resume_dir, jd_path):
    """Create the sample resumes / job description if they do not exist (fresh Colab runtime)."""
    if not glob.glob(os.path.join(resume_dir, "*.txt")):
        os.makedirs(resume_dir, exist_ok=True)
        for fname, text in SAMPLE_RESUMES.items():
            open(os.path.join(resume_dir, fname), "w", encoding="utf-8").write(text)
        print(f"[info] sample resumes written to {resume_dir}/ - replace them with your own TXT files if needed")
    if not os.path.exists(jd_path):
        os.makedirs(os.path.dirname(jd_path) or ".", exist_ok=True)
        open(jd_path, "w", encoding="utf-8").write(SAMPLE_JD)


# ======================================================================
# MAIN PROGRAM
# ======================================================================


import numpy as np
import pandas as pd


W = 60
DEFAULT_QUERY = "Python machine learning data analyst"


def banner(title):
    print("\n" + "=" * W + f"\n{title}\n" + "=" * W)


def section(title):
    print("\n" + "-" * W + f"\n{title}\n" + "-" * W)


def load_resumes(folder):
    files = glob.glob(os.path.join(folder, "*.txt"))
    files.sort(key=lambda f: [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(f))])
    texts = []
    for f in files:
        try:
            with open(f, encoding="utf-8", errors="ignore") as fh:
                t = fh.read()
            if t.strip():
                texts.append((os.path.basename(f), t))
            else:
                print(f"[warning] {f} is empty - skipped")
        except OSError as e:
            print(f"[warning] could not read {f}: {e}")
    return texts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--resumes", default="data/resumes")
    ap.add_argument("--jd", default="data/job_description.txt")
    ap.add_argument("--query", default=DEFAULT_QUERY)
    ap.add_argument("--csv", default="candidate_analysis.csv")
    ap.add_argument("--dendrogram", default="dendrogram.png")
    args, _ = ap.parse_known_args(argv)

    # ------------------------------------------------------------------ load data
    ensure_sample_data(args.resumes, args.jd)
    resumes = load_resumes(args.resumes)
    if len(resumes) < 3:
        sys.exit(f"Need at least 3 resumes in '{args.resumes}' (found {len(resumes)}).")
    try:
        jd_text = open(args.jd, encoding="utf-8", errors="ignore").read()
    except OSError as e:
        sys.exit(f"Cannot read job description: {e}")
    labels = [f"Resume {i + 1}" for i in range(len(resumes))]
    jd_title = jd_text.strip().splitlines()[0]

    banner("AUTOMATED RESUME SCREENER")
    print(f"Job Description : {jd_title}\nNumber of Resumes: {len(resumes)}")

    # ------------------------------------------------------------------ 1. preprocessing + extraction
    docs = [preprocess_document(t) for _, t in resumes]
    infos = [extract_candidate_info(d) for d in docs]

    section("0. PREPROCESSING DEMO (Resume 1, one sentence)")
    d0 = docs[0]
    demo = next((s for s in d0["sentences"] if len(s.split()) >= 8), d0["sentences"][0])
    t, st, lm = clean_text(demo)
    print("Normalized  :", demo)
    print("Tokens      :", t[:9])
    print("Stems       :", st[:9])
    print("Lemmas      :", lm[:9])
    print("POS tags    :", pos_tag_text(demo)[:7])
    print("(Tech terms such as C++, C#, .NET, Power BI are kept; 'Machine Learning' becomes the token 'machine_learning'.)")

    section("1. RESUME INFORMATION")
    for lab, info in zip(labels, infos):
        print(f"\n{lab}")
        print(f"  Candidate Name : {info['name']}")
        print(f"  Universities   : {fmt_list(info['universities'])}")
        print(f"  Organizations  : {fmt_list(info['organizations'])}")
        print(f"  Locations      : {fmt_list(info['locations'])}")
        print(f"  Dates          : {fmt_list(info['dates'])}")
        print(f"  Job Titles     : {fmt_list(info['job_titles'])}")
        print(f"  Detected Skills: {fmt_list(info['skills'])}")
        print(f"  Important Nouns: {fmt_list(info['nouns'], 8)}")

    # ------------------------------------------------------------------ 2. representation
    clean_docs = [d["clean_text"] for d in docs]
    jd_doc = preprocess_document(jd_text)
    bow_vec, bow_m = build_bow(clean_docs)
    tfidf_vec, tfidf_m = build_tfidf(clean_docs)
    sentences = [s for d in docs for s in d["sentence_tokens"]] + jd_doc["sentence_tokens"]
    w2v = train_word2vec([s for s in sentences if s])

    section("2. TEXT REPRESENTATION")
    print("A) Bag of Words (document-term matrix)")
    print(f"   Vocabulary size : {len(bow_vec.vocabulary_)}\n   Matrix shape    : {bow_m.shape}")
    print("   Preview (3 resumes x 6 terms):")
    print("\n".join("   " + l for l in bow_preview(bow_vec, bow_m).to_string().splitlines()))
    print("\nB) TF-IDF")
    print(f"   Matrix shape    : {tfidf_m.shape}")
    print("   Top informative terms per resume:")
    for i, lab in enumerate(labels):
        terms = ", ".join(f"{t} ({w:.2f})" for t, w in top_tfidf_terms(tfidf_vec, tfidf_m, i, 5))
        print(f"   {lab:<10}: {terms}")
    print("\nC) Word2Vec (trained on resume sentences; tiny corpus -> indicative only)")
    print(f"   Vocabulary size : {len(w2v.wv)}   Vector size: {w2v.vector_size}")
    for w in ["python", "machine_learning"]:
        if w in w2v.wv:
            print(f"   Vector('{w}') first 5 dims: {np.round(w2v.wv[w][:5], 3)}")
    pairs = [("pandas", "numpy"), ("machine_learning", "deep_learning"), ("sql", "mysql"),
             ("docker", "kubernetes"), ("python", "javascript")]
    for a, b in pairs:
        s = word_similarity(w2v, a, b)
        if s is not None:
            print(f"   similarity({pretty_token(a)}, {pretty_token(b)}) = {s:.2f}")
    if "python" in w2v.wv:
        near = ", ".join(f"{pretty_token(w)} ({s:.2f})" for w, s in w2v.wv.most_similar("python", topn=4))
        print(f"   Words closest to 'python': {near}")

    # ------------------------------------------------------------------ 3. job matching
    jd_vec = query_vector(tfidf_vec, jd_text)
    tfidf_scores = cosine_scores(jd_vec, tfidf_m)
    bow_scores = cosine_scores(bow_vec.transform([clean_for_tfidf(jd_text)]), bow_m)
    jd_w2v = document_vector(w2v, jd_doc["lemmas"])
    w2v_scores = np.array([
        float(np.dot(jd_w2v, document_vector(w2v, d["lemmas"])) /
              ((np.linalg.norm(jd_w2v) * np.linalg.norm(document_vector(w2v, d["lemmas"]))) or 1))
        for d in docs])

    section("3. JOB MATCHING")
    print("Cosine similarity between each resume and the job description.\n"
          "NOTE: this is TEXTUAL SIMILARITY w.r.t. the supplied job description, not hiring accuracy.\n")
    print(f"{'Rank':<5}{'Candidate':<12}{'Name':<18}{'TF-IDF Similarity':>18}")
    print("-" * 53)
    ranking = rank(tfidf_scores, labels)
    for lab, sc, r in ranking:
        print(f"{r:<5}{lab:<12}{infos[labels.index(lab)]['name']:<18}{sc:>18.3f}")
    print("\nComparison of representations (cosine with job description):")
    print(f"{'Candidate':<12}{'BoW':>8}{'TF-IDF':>9}{'Word2Vec':>10}")
    for i, lab in enumerate(labels):
        print(f"{lab:<12}{bow_scores[i]:>8.3f}{tfidf_scores[i]:>9.3f}{w2v_scores[i]:>10.3f}")
    print("(Word2Vec averages are high for every resume because the corpus is tiny - TF-IDF is used for ranking.)")

    # ------------------------------------------------------------------ 4. skill matching
    jd_skills = extract_skills(jd_doc["normalized"])
    skill_res = [skill_match(jd_skills, info["skills"]) for info in infos]
    section("4. SKILL ANALYSIS")
    print(f"Skills required by the job description: {', '.join(jd_skills)}")
    print("Skill coverage = matched required skills / total required skills x 100.\n"
          "This is a simple rule-based coverage measure, not a professional hiring assessment.")
    for lab, res in zip(labels, skill_res):
        print(f"\n{lab} ({infos[labels.index(lab)]['name']})")
        print(f"  Matched   : {fmt_list(res['matched'])}")
        print(f"  Missing   : {fmt_list(res['missing'])}")
        print(f"  Additional: {fmt_list(res['additional'])}")
        print(f"  Skill Match Percentage: {res['percentage']:.1f}%")

    # ------------------------------------------------------------------ 5. information retrieval
    section("5. INFORMATION RETRIEVAL")
    print("Pipeline: query -> preprocessing -> TF-IDF vector -> cosine similarity vs resumes -> ranking")
    for q in (jd_title, args.query):
        print(f"\nQuery: \"{q}\"   (preprocessed: {clean_for_tfidf(q)})")
        for lab, sc, r in retrieve(q, tfidf_vec, tfidf_m, labels, top_k=5):
            print(f"  {r}. {lab:<10}{infos[labels.index(lab)]['name']:<18}score = {sc:.3f}")

    # ------------------------------------------------------------------ 6. TextRank summary
    section("6. TEXTRANK SUMMARY")
    for lab, d in zip(labels, docs):
        print(f"\nCandidate: {lab} ({infos[labels.index(lab)]['name']})")
        print(f"TextRank Summary:\n  \"{textrank_summary(d['sentences'], 2)}\"")

    # ------------------------------------------------------------------ 7. hierarchical clustering
    hc, best_k, best_s, k_scores = hierarchical_clustering(tfidf_m, labels, args.dendrogram)
    hc_kw = cluster_keywords(tfidf_m, tfidf_vec, hc)
    section("7. HIERARCHICAL CLUSTERING")
    print("Method: cosine distance + average linkage on TF-IDF vectors")
    if k_scores:
        print("Silhouette score per number of clusters: " + ", ".join(f"k={k}: {s:.3f}" for k, s in k_scores.items()))
        print(f"Chosen k = {best_k} (highest silhouette = {best_s:.3f}); an internal metric, not proof of 'correct' groups.")
    print()
    for lab, c in zip(labels, hc):
        print(f"{lab:<10} -> Cluster {c}   ({infos[labels.index(lab)]['name']})")
    for c, kw in hc_kw.items():
        print(f"Cluster {c} keywords: {kw}")
    print(f"\nDendrogram saved as: {args.dendrogram}")

    # ------------------------------------------------------------------ 8. DBSCAN
    db = dbscan_clustering(tfidf_m)
    db_kw = cluster_keywords(tfidf_m, tfidf_vec, db)
    section("8. DBSCAN (eps=0.88, min_samples=2, cosine distance)")
    for lab, c in zip(labels, db):
        tag = "  <- noise / outlier" if c == -1 else ""
        print(f"{lab:<10} -> Cluster {c}{tag}")
    for c, kw in db_kw.items():
        if c != -1:
            print(f"Cluster {c} keywords: {kw}")
    noise = [l for l, c in zip(labels, db) if c == -1]
    print("Noise/outliers (label -1):", ", ".join(noise) if noise else "none")

    # ------------------------------------------------------------------ 9. topic modelling
    topics, dominant = nmf_topics(tfidf_m, tfidf_vec)
    section("9. TOPIC MODELLING (NMF on TF-IDF)")
    for i, words in enumerate(topics, 1):
        members = [labels[j] for j in range(len(labels)) if dominant[j] == i - 1]
        print(f"Topic {i}: {', '.join(words)}\n         dominant in: {', '.join(members) if members else '-'}")

    # ------------------------------------------------------------------ final table + CSV
    section("FINAL CANDIDATE COMPARISON")
    rows = []
    for i, lab in enumerate(labels):
        rows.append({
            "Candidate": lab, "Name": infos[i]["name"],
            "University": infos[i]["universities"][0] if infos[i]["universities"] else NOT_DETECTED,
            "Job_Title": infos[i]["job_titles"][0] if infos[i]["job_titles"] else NOT_DETECTED,
            "TFIDF_Similarity": round(float(tfidf_scores[i]), 4),
            "Skill_Match_Percentage": round(skill_res[i]["percentage"], 1),
            "Matched_Skills": "; ".join(skill_res[i]["matched"]),
            "Missing_Skills": "; ".join(skill_res[i]["missing"]),
            "Cluster": int(hc[i]), "DBSCAN_Cluster": int(db[i]),
        })
    df = pd.DataFrame(rows).sort_values("TFIDF_Similarity", ascending=False)
    print(f"{'Candidate':<11}{'Name':<17}{'Similarity':>11}{'Skill Match %':>15}{'Cluster':>9}{'DBSCAN':>8}")
    for _, r in df.iterrows():
        print(f"{r.Candidate:<11}{r.Name:<17}{r.TFIDF_Similarity:>11.3f}{r.Skill_Match_Percentage:>14.1f}%{r.Cluster:>9}{r.DBSCAN_Cluster:>8}")
    try:
        df.to_csv(args.csv, index=False)
        print(f"\nSaved: {args.csv}, {args.dendrogram}")
    except OSError as e:
        print(f"[warning] could not write CSV: {e}")
    print("\nDisclaimer: scores show textual similarity and keyword coverage only; they are not a hiring decision.")

    # show the dendrogram inside Colab / Jupyter if possible
    try:
        from IPython.display import Image, display
        display(Image(args.dendrogram))
    except Exception:
        pass


if __name__ == "__main__":
    main()
