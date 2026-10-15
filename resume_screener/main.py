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

import numpy as np
import pandas as pd

from modules.preprocessing import preprocess_document, pretty_token, clean_for_tfidf, clean_text, pos_tag_text
from modules.extraction import extract_candidate_info, extract_skills, fmt_list, NOT_DETECTED
from modules.representation import build_bow, bow_preview, build_tfidf, top_tfidf_terms, train_word2vec, word_similarity, document_vector
from modules.matching import query_vector, cosine_scores, rank, skill_match, retrieve
from modules.summarization import textrank_summary
from modules.clustering import hierarchical_clustering, dbscan_clustering, cluster_keywords
from modules.topic_modeling import nmf_topics

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
