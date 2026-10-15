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

from modules.preprocessing import pretty_token


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
