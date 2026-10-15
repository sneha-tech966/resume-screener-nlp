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

from modules.preprocessing import clean_for_tfidf


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
