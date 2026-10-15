"""
MODULE 9 - TOPIC MODELLING (NMF)
================================
NMF factorizes the TF-IDF matrix V (resumes x words) into W (resumes x topics) and
H (topics x words). Each row of H is a topic = a group of words that appear together.
NMF is used instead of LDA because it is stable and works well on very small corpora.
"""
import numpy as np
from sklearn.decomposition import NMF

from modules.preprocessing import pretty_token


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
