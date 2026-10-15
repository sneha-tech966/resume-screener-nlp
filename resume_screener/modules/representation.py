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

from modules.preprocessing import pretty_token


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
