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

from modules.preprocessing import clean_for_tfidf


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
