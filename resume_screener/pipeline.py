"""
PIPELINE (used by the web UI)
=============================
Runs every NLP step on a list of resumes and returns ONE dictionary with all intermediate
results, so the UI can show every step. It only calls the same module functions that the
terminal version (main.py) uses - no NLP logic is duplicated here.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD

from modules.preprocessing import preprocess_document, clean_for_tfidf
from modules.extraction import extract_candidate_info, extract_skills
from modules.representation import (build_bow, build_tfidf, train_word2vec, document_vector)
from modules.matching import query_vector, cosine_scores, rank, skill_match
from modules.summarization import textrank_summary
from modules.integrity import build_integrity
from modules.clustering import hierarchical_clustering, dbscan_clustering, cluster_keywords
from modules.topic_modeling import nmf_topics

MIN_FOR_CLUSTERING = 3      # silhouette / DBSCAN / NMF need at least 3 documents


def _cos(a, b):
    d = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / d) if d else 0.0


def run_pipeline(files, jd_text, n_summary=2, dendrogram_path="dendrogram.png", progress=None):
    """files: list of dicts {"filename": str, "text": str, "info": dict}"""
    step = progress or (lambda msg: None)
    R = {"n": len(files), "files": files}

    step("Preprocessing resumes (tokenize, stop-words, stem, lemmatize, POS-tag)")
    R["docs"] = [preprocess_document(f["text"]) for f in files]
    R["jd_doc"] = preprocess_document(jd_text)
    R["jd_text"] = jd_text
    R["jd_title"] = jd_text.strip().splitlines()[0] if jd_text.strip() else "Job"

    step("Extracting entities, skills and nouns (spaCy NER + POS + dictionary)")
    R["infos"] = [extract_candidate_info(d) for d in R["docs"]]
    labels, seen = [], {}
    for f, info in zip(files, R["infos"]):
        base = info["name"] if info["name"] != "Not detected" else f["filename"].rsplit(".", 1)[0]
        seen[base] = seen.get(base, 0) + 1
        labels.append(base if seen[base] == 1 else f"{base} ({seen[base]})")
    R["labels"] = labels

    step("Building Bag-of-Words, TF-IDF and Word2Vec representations")
    clean_docs = [d["clean_text"] for d in R["docs"]]
    R["bow_vec"], R["bow_m"] = build_bow(clean_docs)
    R["tfidf_vec"], R["tfidf_m"] = build_tfidf(clean_docs)
    sents = [s for d in R["docs"] for s in d["sentence_tokens"]] + R["jd_doc"]["sentence_tokens"]
    R["w2v"] = train_word2vec([s for s in sents if s])

    step("Matching resumes with the job description (cosine similarity)")
    R["jd_vec"] = query_vector(R["tfidf_vec"], jd_text)
    R["tfidf_scores"] = cosine_scores(R["jd_vec"], R["tfidf_m"])
    R["bow_scores"] = cosine_scores(R["bow_vec"].transform([clean_for_tfidf(jd_text)]), R["bow_m"])
    jd_w = document_vector(R["w2v"], R["jd_doc"]["lemmas"])
    R["w2v_scores"] = np.array([_cos(jd_w, document_vector(R["w2v"], d["lemmas"])) for d in R["docs"]])
    R["ranking"] = rank(R["tfidf_scores"], list(range(len(files))))      # (index, score, rank)

    step("Comparing skills with the job description")
    R["jd_skills"] = extract_skills(R["jd_doc"]["normalized"])
    R["skill_res"] = [skill_match(R["jd_skills"], i["skills"]) for i in R["infos"]]

    step("Checking resumes for hidden text and keyword stuffing")
    R["integrity"] = []
    for i, f in enumerate(files):
        rep = build_integrity(f["info"], f["text"], R["infos"][i]["skills"], R["jd_skills"])
        if rep["hidden_text"]:
            # what would the score have been if the hidden text had been trusted?
            full = f["text"] + "\n" + rep["hidden_text"]
            v = query_vector(R["tfidf_vec"], full)
            naive_sim = float(cosine_scores(R["jd_vec"], v)[0])
            naive_skills = extract_skills(full)
            naive_cov = skill_match(R["jd_skills"], naive_skills)["percentage"]
            others = [s for j, s in enumerate(R["tfidf_scores"]) if j != i]
            rep["naive"] = {"similarity": naive_sim, "coverage": naive_cov,
                            "rank": 1 + sum(1 for s in others if s > naive_sim)}
        R["integrity"].append(rep)

    step("Summarising resumes with TextRank")
    R["summaries"] = [textrank_summary(d["sentences"], n_summary) for d in R["docs"]]

    R["can_cluster"] = len(files) >= MIN_FOR_CLUSTERING
    if R["can_cluster"]:
        step("Clustering resumes (hierarchical + DBSCAN) and modelling topics")
        hc, k, sil, k_scores = hierarchical_clustering(R["tfidf_m"], labels, dendrogram_path)
        R.update(hc=hc, hc_k=k, hc_sil=sil, hc_k_scores=k_scores, dendrogram_path=dendrogram_path,
                 hc_keywords=cluster_keywords(R["tfidf_m"], R["tfidf_vec"], hc))
        R["db"] = dbscan_clustering(R["tfidf_m"])
        R["db_keywords"] = cluster_keywords(R["tfidf_m"], R["tfidf_vec"], R["db"])
        R["topics"], R["dominant_topic"] = nmf_topics(R["tfidf_m"], R["tfidf_vec"])
        R["xy"] = TruncatedSVD(2, random_state=42).fit_transform(R["tfidf_m"])   # 2-D map for plots
    step("Done")
    return R
