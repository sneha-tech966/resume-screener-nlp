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
