"""
AUTOMATED RESUME SCREENER - WEB UI (Streamlit)
Run:  streamlit run app.py
Upload one or many PDF resumes + a job description, press "Run analysis", and every NLP step
is shown in order, followed by the final results.
"""
import glob
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Automated Resume Screener", page_icon="📄", layout="wide")
st.title("📄 Automated Resume Screener")
_loading = st.info("⏳ Starting up - the first launch can take up to a minute while components are prepared.")

import importlib.util, subprocess, sys                       # noqa: E401
for _mod, _pip in (("pypdf", "pypdf"), ("pymupdf", "pymupdf")):   # packages needed only by the web UI
    if importlib.util.find_spec(_mod) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", _pip],
                              env=dict(os.environ, PIP_BREAK_SYSTEM_PACKAGES="1"))
import main as _bootstrap_loader      # noqa: F401  (installs packages / NLTK data / spaCy model once)
from modules.pdf_reader import extract_text
from modules.preprocessing import clean_for_tfidf, clean_text, pos_tag_text, pretty_token
from modules.representation import bow_preview, top_tfidf_terms, word_similarity
from modules.matching import retrieve
from modules.clustering import dbscan_clustering, cluster_keywords
from modules.topic_modeling import nmf_topics
from pipeline import run_pipeline, MIN_FOR_CLUSTERING

_loading.empty()

ROOT = os.path.dirname(os.path.abspath(__file__))
SAMPLE_RESUME_DIR = os.path.join(ROOT, "data", "resumes_pdf")
SAMPLE_JD = os.path.join(ROOT, "data", "job_description.txt")
TRUST_TEST_DIR = os.path.join(ROOT, "data", "resumes_pdf_trust_test")

# ----------------------------------------------------------------------------- CSS & helpers
st.markdown("""
<style>
.chip {
    display: inline-block;
    padding: 3px 10px;
    margin: 2px 4px 2px 0;
    border-radius: 12px;
    font-size: 0.82rem;
    font-weight: 500;
    line-height: 1.4;
    border: 1px solid transparent;
}
.chip.ok {
    background-color: rgba(30, 142, 62, 0.18);
    color: #1b873f;
    border-color: rgba(30, 142, 62, 0.35);
}
.chip.bad {
    background-color: rgba(217, 48, 37, 0.18);
    color: #d93025;
    border-color: rgba(217, 48, 37, 0.35);
}
.chip.warn {
    background-color: rgba(227, 116, 0, 0.18);
    color: #c96800;
    border-color: rgba(227, 116, 0, 0.35);
}
.chip.info {
    background-color: rgba(26, 115, 232, 0.18);
    color: #1a73e8;
    border-color: rgba(26, 115, 232, 0.35);
}
.chip.gray {
    background-color: rgba(128, 134, 139, 0.18);
    color: #5f6368;
    border-color: rgba(128, 134, 139, 0.35);
}
@media (prefers-color-scheme: dark) {
    .chip.ok { color: #81c995; border-color: rgba(129, 201, 149, 0.4); }
    .chip.bad { color: #f28b82; border-color: rgba(242, 139, 130, 0.4); }
    .chip.warn { color: #fdd663; border-color: rgba(253, 214, 99, 0.4); }
    .chip.info { color: #8ab4f8; border-color: rgba(138, 180, 248, 0.4); }
    .chip.gray { color: #dadce0; border-color: rgba(218, 220, 224, 0.4); }
}
[data-testid="stAppViewContainer"][data-theme="dark"] .chip.ok,
[data-theme="dark"] .chip.ok { color: #81c995; border-color: rgba(129, 201, 149, 0.4); }
[data-theme="dark"] .chip.bad { color: #f28b82; border-color: rgba(242, 139, 130, 0.4); }
[data-theme="dark"] .chip.warn { color: #fdd663; border-color: rgba(253, 214, 99, 0.4); }
[data-theme="dark"] .chip.info { color: #8ab4f8; border-color: rgba(138, 180, 248, 0.4); }
[data-theme="dark"] .chip.gray { color: #dadce0; border-color: rgba(218, 220, 224, 0.4); }

.stepnote {
    font-size: 0.9rem;
    opacity: 0.85;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)


def chips(items, kind="info"):
    if not items:
        return "<span class='chip gray'>Not detected</span>"
    return "".join(f"<span class='chip {kind}'>{i}</span>" for i in items)


def step_header(num, title, note):
    st.markdown("---")
    st.subheader(f"Step {num} - {title}")
    st.markdown(f"<div class='stepnote'>{note}</div>", unsafe_allow_html=True)


def _setup_fig_theme(fig, ax):
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    text_color = "#31333F"
    tick_color = "#4A4D59"
    ax.tick_params(colors=tick_color, which="both")
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    for sp in ax.spines.values():
        sp.set_color(tick_color)


def hbar(labels, values, title, xlabel="", fmt="{:.3f}", color="#1a73e8"):
    fig, ax = plt.subplots(figsize=(7, max(1.6, 0.42 * len(labels) + 0.8)))
    _setup_fig_theme(fig, ax)
    ypos = np.arange(len(labels))
    ax.barh(ypos, values, color=color)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=11)
    for y, v in zip(ypos, values):
        ax.text(v, y, " " + fmt.format(v), va="center", fontsize=8, color="#31333F")
    ax.set_xlim(0, max(values) * 1.18 if len(values) and max(values) > 0 else 1)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)


def scatter(xy, labels, assignment, title):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _setup_fig_theme(fig, ax)
    cmap = plt.get_cmap("tab10")
    for c in sorted(set(assignment)):
        m = assignment == c
        if c == -1:
            ax.scatter(xy[m, 0], xy[m, 1], c="#9aa0a6", marker="x", s=70, label="noise (-1)")
        else:
            ax.scatter(xy[m, 0], xy[m, 1], color=cmap(int(c) % 10), s=80, label=f"Cluster {c}")
    for (x, y), lab in zip(xy, labels):
        ax.annotate(lab.split()[0], (x, y), fontsize=8, xytext=(4, 4), textcoords="offset points", color="#31333F")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("component 1")
    ax.set_ylabel("component 2")
    leg = ax.legend(fontsize=8, framealpha=0.2)
    for text in leg.get_texts():
        text.set_color("#31333F")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig, transparent=True)
    plt.close(fig)


def read_uploaded(file):
    text, info = extract_text(file.getvalue(), file.name)
    return {"filename": file.name, "text": text, "info": info}


def read_path(path):
    with open(path, "rb") as fh:
        text, info = extract_text(fh.read(), os.path.basename(path))
    return {"filename": os.path.basename(path), "text": text, "info": info}


def natural_key(p):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", os.path.basename(p))]



# ============================================================================= technical view
def render_technical(R):
    """Developer / evaluator view: every NLP step with its intermediate output."""
    labels, infos, docs, n = R["labels"], R["infos"], R["docs"], R["n"]
    # ----------------------------------------------------------------------------- step 0 input
    step_header(0, "Input check: text extracted from the PDFs",
                "The text layer of each PDF is extracted and repaired (bullets, wrapped lines, ALL-CAPS headings).")
    rows = [{"File": f["filename"], "Pages": f["info"].get("pages", 1), "Characters": f["info"].get("chars", 0),
             "Words": len(f["text"].split()), "Status": "OK" if not f["info"]["warnings"] else " ".join(f["info"]["warnings"])}
            for f in R["files"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True)
    pick = st.selectbox("Show extracted text of", range(n), format_func=lambda i: R["files"][i]["filename"], key="s0")
    st.text_area("Extracted text", R["files"][pick]["text"], height=220, disabled=True)

    # ----------------------------------------------------------------------------- step 1 preprocessing
    step_header(1, "Text preprocessing", "Lowercasing, URL/e-mail removal, tokenization, stop-word removal, stemming, "
                "lemmatization and POS tagging. Technical terms (C++, .NET, Machine Learning) are protected.")
    pick = st.selectbox("Resume", range(n), format_func=lambda i: labels[i], key="s1")
    d = docs[pick]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Words in raw text", len(R["files"][pick]["text"].split()))
    m2.metric("Tokens after stop-word removal", len(d["tokens"]))
    m3.metric("Unique lemmas", len(set(d["lemmas"])))
    m4.metric("Sentences", len(d["sentences"]))
    a, b = st.columns(2)
    with a:
        st.markdown("**Normalized text** (URLs, e-mails, phones removed; case & punctuation kept for NER)")
        st.text_area("normalized", d["normalized"][:1500], height=230, disabled=True, label_visibility="collapsed")
    with b:
        st.markdown("**Clean text for vectors** (lowercase, stop-words removed, lemmatized, skills as single tokens)")
        st.text_area("clean", d["clean_text"][:1500], height=230, disabled=True, label_visibility="collapsed")
    demo = next((s for s in d["sentences"] if len(s.split()) >= 8), d["sentences"][0] if d["sentences"] else "")
    if demo:
        st.markdown(f"**Worked example sentence:** _{demo}_")
        toks, stems, lems = clean_text(demo)
        t1, t2 = st.columns(2)
        with t1:
            st.markdown("Tokens → Stems → Lemmas")
            st.dataframe(pd.DataFrame({"Token": toks, "Stem": stems, "Lemma": lems}), hide_index=True)
        with t2:
            st.markdown("POS tags (on the normalized sentence)")
            st.dataframe(pd.DataFrame(pos_tag_text(demo), columns=["Word", "POS tag"]), hide_index=True)
        st.caption("NN/NNP = noun / proper noun, JJ = adjective, VB* = verb, IN = preposition. Stems are cruder "
                   "('science' → 'scienc'), so lemmas are used for TF-IDF.")

    # ----------------------------------------------------------------------------- step 2 extraction
    step_header(2, "Information extraction", "spaCy NER finds organisations, places and dates; rules find the name, "
                "university and job title; a skill dictionary finds technologies; POS tags give important nouns.")
    tbl = pd.DataFrame([{
        "Candidate": labels[i], "File": R["files"][i]["filename"],
        "University": ", ".join(x["universities"]) or "Not detected",
        "Organizations": ", ".join(x["organizations"]) or "Not detected",
        "Locations": ", ".join(x["locations"]) or "Not detected",
        "Job titles": ", ".join(x["job_titles"]) or "Not detected",
        "Dates": ", ".join(x["dates"]) or "Not detected",
        "# Skills": len(x["skills"])} for i, x in enumerate(infos)])
    st.dataframe(tbl, hide_index=True)
    for i, x in enumerate(infos):
        with st.expander(f"{labels[i]} - skills and important nouns"):
            st.markdown("**Detected skills**", help="Dictionary-based, case-insensitive, multi-word aware")
            st.markdown(chips(x["skills"], "info"), unsafe_allow_html=True)
            st.markdown("**Important nouns / technical terms** (POS-based)")
            st.markdown(chips(x["nouns"], "gray"), unsafe_allow_html=True)

    # ----------------------------------------------------------------------------- step 3 representation
    step_header(3, "Text representation", "Turning text into numbers: Bag of Words (counts), TF-IDF (weighted counts), "
                "Word2Vec (learned word meaning).")
    tab_b, tab_t, tab_w = st.tabs(["Bag of Words", "TF-IDF", "Word2Vec"])
    with tab_b:
        c1, c2 = st.columns(2)
        c1.metric("Vocabulary size", len(R["bow_vec"].vocabulary_))
        c2.metric("Matrix shape (resumes × terms)", str(R["bow_m"].shape))
        st.markdown("Preview of the document-term matrix (first 6 vocabulary terms, up to 3 resumes):")
        prev = bow_preview(R["bow_vec"], R["bow_m"], 6, min(3, n))
        prev.index = labels[:len(prev)]
        st.dataframe(prev)
        st.caption("Every word counts the same - common words are as important as rare ones.")
    with tab_t:
        st.metric("Matrix shape (resumes × terms)", str(R["tfidf_m"].shape))
        pick = st.selectbox("Top TF-IDF terms of", range(n), format_func=lambda i: labels[i], key="s3")
        top = top_tfidf_terms(R["tfidf_vec"], R["tfidf_m"], pick, 10)
        hbar([t for t, _ in top], [w for _, w in top], f"Most informative terms - {labels[pick]}", "TF-IDF weight", "{:.2f}")
        if n == 1:
            st.info("With a single resume the IDF part cannot discriminate between terms (every term appears in 1 of 1 "
                    "documents). Upload several resumes for a meaningful TF-IDF.")
        st.caption("High weight = frequent in this resume but rare in the others, i.e. what makes the candidate distinctive.")
    with tab_w:
        w2v = R["w2v"]
        c1, c2 = st.columns(2)
        c1.metric("Word2Vec vocabulary", len(w2v.wv))
        c2.metric("Vector size", w2v.vector_size)
        pairs = [("pandas", "numpy"), ("machine_learning", "deep_learning"), ("sql", "mysql"),
                 ("docker", "kubernetes"), ("python", "javascript"), ("react", "nodedotjs")]
        prow = [{"Word 1": pretty_token(a), "Word 2": pretty_token(b), "Similarity": round(s, 3)}
                for a, b in pairs if (s := word_similarity(w2v, a, b)) is not None]
        if prow:
            st.markdown("Similarity between related terms:")
            st.dataframe(pd.DataFrame(prow), hide_index=True)
        word = st.text_input("Find words closest to (e.g. python, docker, sql)", "python").strip().lower().replace(" ", "_")
        if word in w2v.wv:
            near = w2v.wv.most_similar(word, topn=6)
            st.dataframe(pd.DataFrame([(pretty_token(w), round(s, 3)) for w, s in near], columns=["Word", "Similarity"]),
                         hide_index=True)
            st.caption(f"First 5 numbers of the vector for '{word}': {np.round(w2v.wv[word][:5], 3)}")
        elif word:
            st.warning(f"'{word}' is not in the vocabulary.")
        st.caption("Trained only on the uploaded resumes, so it is a demonstration of the idea; with few resumes the "
                   "numbers are noisy.")

    # ----------------------------------------------------------------------------- step 4 job matching
    step_header(4, "Job description matching", "The job description is converted to a TF-IDF vector and compared with "
                "every resume using cosine similarity (1 = identical wording, 0 = nothing in common).")
    order = [i for i, _, _ in R["ranking"]]
    hbar([labels[i] for i in order], [R["tfidf_scores"][i] for i in order], "TF-IDF cosine similarity with the job description",
         "similarity")
    cmp_df = pd.DataFrame({"Candidate": labels, "Bag of Words": R["bow_scores"].round(3),
                           "TF-IDF (used for ranking)": R["tfidf_scores"].round(3), "Word2Vec (avg. vectors)": R["w2v_scores"].round(3)})
    st.markdown("Comparison of the three representations:")
    st.dataframe(cmp_df.iloc[order], hide_index=True)
    st.caption("This is textual similarity with respect to the supplied job description, **not hiring accuracy**. "
               "Word2Vec averages are high for everyone on a small corpus, which is why TF-IDF is used for ranking.")

    # ----------------------------------------------------------------------------- step 5 skills
    step_header(5, "Skill analysis", "Skills required by the job description are compared with each resume. "
                "Coverage = matched required skills ÷ total required skills × 100 (a simple rule-based measure).")
    st.markdown("**Required skills (from the job description):**")
    st.markdown(chips(R["jd_skills"], "info"), unsafe_allow_html=True)
    if not R["jd_skills"]:
        st.warning("No known skills were found in the job description, so skill coverage is 0 for everyone.")
    mat = pd.DataFrame({s: ["✅" if s in R["skill_res"][i]["matched"] else "❌" for i in range(n)] for s in R["jd_skills"]},
                       index=labels)
    mat["Coverage %"] = [round(r["percentage"], 1) for r in R["skill_res"]]
    st.dataframe(mat, column_config={"Coverage %": st.column_config.ProgressColumn("Coverage %", min_value=0, max_value=100,
                                                                                 format="%.1f%%")})
    for i in range(n):
        r = R["skill_res"][i]
        with st.expander(f"{labels[i]} - {r['percentage']:.1f}% coverage"):
            st.markdown("**Matched**")
            st.markdown(chips(r["matched"], "ok"), unsafe_allow_html=True)
            st.markdown("**Missing**")
            st.markdown(chips(r["missing"], "bad") if r["missing"] else "<span class='chip ok'>Nothing missing</span>",
                        unsafe_allow_html=True)
            st.markdown("**Additional skills**")
            st.markdown(chips(r["additional"], "info"), unsafe_allow_html=True)
    st.caption("Rule-based keyword coverage - not a professional hiring assessment.")

    # ----------------------------------------------------------------------------- step 6 IR
    step_header(6, "Information retrieval (search)", "The query is treated like a search query: preprocess → TF-IDF vector → "
                "cosine similarity with every resume → ranking. Try your own query.")
    query = st.text_input("Search query", "Python machine learning data analyst")
    for q in (R["jd_title"], query):
        if q.strip():
            st.markdown(f"**Query:** “{q}”  &nbsp; _preprocessed → `{clean_for_tfidf(q)}`_")
            res = retrieve(q, R["tfidf_vec"], R["tfidf_m"], list(range(n)), top_k=min(5, n))
            st.dataframe(pd.DataFrame([{"Rank": r, "Candidate": labels[i], "Score": round(s, 3)} for i, s, r in res]),
                         hide_index=True)

    # ----------------------------------------------------------------------------- step 7 summaries
    step_header(7, "TextRank summaries", "Sentences become graph nodes (edges = TF-IDF cosine similarity); PageRank picks the "
                "most central sentences, shown in original order.")
    for i in range(n):
        with st.container(border=True):
            st.markdown(f"**{labels[i]}**")
            st.write(R["summaries"][i])

    # ----------------------------------------------------------------------------- steps 8-10
    if not R["can_cluster"]:
        for num, title in ((8, "Hierarchical clustering"), (9, "DBSCAN"), (10, "Topic modelling")):
            step_header(num, title, "")
            st.info(f"Needs at least {MIN_FOR_CLUSTERING} resumes - you uploaded {n}. Upload more to see this step.")
    else:
        hc, db = R["hc"], R["db"]
        step_header(8, "Hierarchical clustering", "Resumes are merged bottom-up using cosine distance (average linkage). "
                    "The number of clusters is the one with the best silhouette score.")
        c1, c2 = st.columns(2)
        c1.metric("Chosen number of clusters", R["hc_k"])
        c2.metric("Silhouette score", "n/a" if R["hc_sil"] is None else f"{R['hc_sil']:.3f}")
        if R["hc_k_scores"]:
            st.caption("Silhouette per k: " + ", ".join(f"k={k}: {s:.3f}" for k, s in R["hc_k_scores"].items()) +
                       ". Low values mean clusters overlap - this is an internal metric, not proof the groups are 'correct'.")
        st.image(R["dendrogram_path"], caption="Dendrogram")
        ca, cb = st.columns([1, 1])
        with ca:
            st.dataframe(pd.DataFrame({"Candidate": labels, "Cluster": hc,
                                       "Cluster keywords": [R["hc_keywords"][c] for c in hc]}), hide_index=True)
        with cb:
            scatter(R["xy"], labels, hc, "2-D map of resumes (SVD of TF-IDF), coloured by cluster")

        step_header(9, "DBSCAN (density clustering & outliers)", "Resumes with at least 'min samples' neighbours within distance "
                    "'eps' form a cluster; the rest are noise (-1) = unusual resumes.")
        e1, e2 = st.columns(2)
        eps = e1.slider("eps (max cosine distance between neighbours)", 0.50, 1.00, 0.88, 0.01)
        ms = e2.slider("min samples", 2, 4, 2)
        db = dbscan_clustering(R["tfidf_m"], eps=eps, min_samples=ms)
        kw = cluster_keywords(R["tfidf_m"], R["tfidf_vec"], db)
        ca, cb = st.columns([1, 1])
        with ca:
            st.dataframe(pd.DataFrame({"Candidate": labels, "DBSCAN cluster": db,
                                       "Meaning": ["noise / outlier" if c == -1 else kw[c] for c in db]}), hide_index=True)
            out = [labels[i] for i, c in enumerate(db) if c == -1]
            st.markdown("**Outliers:** " + (", ".join(out) if out else "none"))
        with cb:
            scatter(R["xy"], labels, db, "DBSCAN result")
        st.caption("Small eps → almost everything is noise; large eps → everything merges into one cluster.")

        step_header(10, "Topic modelling (NMF)", "NMF splits the TF-IDF matrix into topics (groups of words that occur "
                    "together) and tells which topic dominates each resume.")
        max_t = min(6, n - 1)
        nt = st.slider("Number of topics", 2, max_t, min(4, max_t)) if max_t > 2 else 2
        topics, dom = nmf_topics(R["tfidf_m"], R["tfidf_vec"], n_topics=nt)
        for t, words in enumerate(topics):
            who = [labels[i] for i in range(n) if dom[i] == t]
            with st.container(border=True):
                st.markdown(f"**Topic {t + 1}**")
                st.markdown(chips(words, "info"), unsafe_allow_html=True)
                st.caption("Dominant in: " + (", ".join(who) if who else "-"))



# ============================================================================= sidebar
st.caption("NLP-based resume screening with an authenticity check for hidden text and keyword stuffing. "
           "Scores show **textual similarity** and **keyword skill coverage** for the supplied job description - "
           "they are not a hiring decision.")

sample_resumes = sorted(glob.glob(os.path.join(SAMPLE_RESUME_DIR, "*.pdf")), key=natural_key)
trust_tests = sorted(glob.glob(os.path.join(TRUST_TEST_DIR, "*.pdf")), key=natural_key)

with st.sidebar:
    st.header("Job description")
    jd_mode = st.radio("Source", ["Sample (ML Intern)", "Paste text", "Upload PDF / TXT"], label_visibility="collapsed")
    jd_text, jd_warn = "", []
    if jd_mode.startswith("Sample"):
        jd_text = open(SAMPLE_JD, encoding="utf-8").read()
        with st.expander("View sample job description"):
            st.text(jd_text)
    elif jd_mode == "Paste text":
        jd_text = st.text_area("Paste the job description (first line = job title)", height=250)
    else:
        up = st.file_uploader("Job description file", type=["pdf", "txt"], key="jd_file")
        if up:
            jd_text, info = extract_text(up.getvalue(), up.name)
            jd_warn = info["warnings"]
    for w in jd_warn:
        st.warning(w)

    st.header("Options")
    n_summary = st.slider("Sentences per candidate summary", 1, 4, 2)
    push_down = st.checkbox("Rank high-risk resumes lower", value=True,
                            help="Resumes with hidden text or clearly unsupported skills are placed below the others "
                                 "in the ranking. They stay visible with the reason shown.")
    show_tech = st.toggle("Show technical NLP details", value=False,
                          help="Adds a section with every intermediate NLP step (preprocessing, vectors, clustering ...).")

# ============================================================================= resume input
st.header("Upload resumes")
c1, c2 = st.columns([3, 1])
with c1:
    uploads = st.file_uploader("Drag and drop one or more resumes (PDF or TXT)", type=["pdf", "txt"],
                               accept_multiple_files=True, key="resumes")
with c2:
    st.write("")
    use_sample = st.checkbox(f"Add the {len(sample_resumes)} sample resumes", value=False, disabled=not sample_resumes,
                             help="Synthetic resumes (fictional people) for a quick demo.")
    use_trust = st.checkbox(f"Add {len(trust_tests)} test resumes with tricks", value=False, disabled=not trust_tests,
                            help="Hidden white text, micro text, visible keyword stuffing, and one honest resume with a "
                                 "dark header (should not be flagged).")

run = st.button("▶ Run analysis", type="primary")

if run:
    files = [read_uploaded(f) for f in (uploads or [])]
    if use_sample:
        files += [read_path(p) for p in sample_resumes]
    if use_trust:
        files += [read_path(p) for p in trust_tests]
    usable = [f for f in files if f["text"].strip()]
    if not files:
        st.error("Please upload at least one resume (or add the sample resumes).")
    elif not jd_text.strip():
        st.error("Please provide a job description in the sidebar.")
    elif not usable:
        st.error("None of the files contained readable text.")
    else:
        for f in files:
            if f not in usable:
                st.warning(f"{f['filename']}: skipped - " + (" ".join(f["info"]["warnings"]) or "no text"))
        with st.spinner("Analysing resumes ..."):
            try:
                st.session_state["R"] = run_pipeline(usable, jd_text, n_summary=n_summary,
                                                     dendrogram_path=os.path.join(ROOT, "dendrogram.png"))
            except Exception as e:                                           # never show a raw traceback to the user
                st.error("The analysis failed. Please check the files and try again.")
                with st.expander("Error details"):
                    st.exception(e)
                st.session_state.pop("R", None)

R = st.session_state.get("R")
if not R:
    st.info("👋 **Welcome to Automated Resume Screener!** Follow these 3 simple steps to get started:")
    c_step1, c_step2, c_step3 = st.columns(3)
    with c_step1:
        st.markdown("### 1. Select Job Description")
        st.caption("Use the default sample ML Intern description in the sidebar, paste your own requirements, or upload a JD PDF/TXT.")
    with c_step2:
        st.markdown("### 2. Add Resumes")
        st.caption("Upload your candidate resumes (PDF or TXT), or tick **Add the sample resumes** and **Add test resumes with tricks**.")
    with c_step3:
        st.markdown("### 3. Run & Screen")
        st.caption("Click **▶ Run analysis** to review ranked scores, skill coverage, authenticity flags, and detailed profiles.")
    st.stop()

labels, infos, n, integ = R["labels"], R["infos"], R["n"], R["integrity"]
high = [i for i in range(n) if integ[i]["trust"].startswith("🚩")]
review = [i for i in range(n) if integ[i]["trust"].startswith("⚠️")]

# ============================================================================= results
st.markdown("---")
st.header(f"Results - {R['jd_title']}", anchor="final-results")
if high:
    st.error(f"🚩 {len(high)} resume(s) show signs of keyword manipulation: "
             + ", ".join(labels[i] for i in high) + ".  See the **Authenticity check** below.")
if review:
    st.warning(f"⚠️ {len(review)} resume(s) should be reviewed manually: " + ", ".join(labels[i] for i in review) + ".")

final = pd.DataFrame({
    "Candidate": labels,
    "University": [", ".join(x["universities"][:1]) or "Not detected" for x in infos],
    "Job title": [", ".join(x["job_titles"][:1]) or "Not detected" for x in infos],
    "Text match": R["tfidf_scores"].round(3),
    "Skill coverage %": [round(r["percentage"], 1) for r in R["skill_res"]],
    "Verified coverage %": [round(g["verified_coverage"], 1) for g in integ],
    "Trust": [g["trust"] for g in integ],
    "Notes": [len(g["flags"]) for g in integ],
})
sort_by = st.radio("Rank candidates by", ["Text match", "Skill coverage %", "Verified coverage %"], horizontal=True,
                   help="Text match = TF-IDF cosine similarity with the job description. Skill coverage = share of "
                        "required skills mentioned. Verified coverage = share of required skills that are also "
                        "backed by experience or projects.")
final["_hr"] = [t.startswith("🚩") for t in final["Trust"]]
final = final.sort_values(["_hr", sort_by] if push_down else [sort_by], ascending=[True, False] if push_down else False)
final = final.drop(columns="_hr").reset_index(drop=True)
final.insert(0, "Rank", range(1, len(final) + 1))

k1, k2, k3, k4 = st.columns(4)
k1.metric("Resumes analysed", n)
k2.metric("Top candidate", final.iloc[0]["Candidate"])
k3.metric("Need attention", len(high) + len(review))
k4.metric("With hidden text", sum(1 for g in integ if g["hidden_text"]))

st.dataframe(final, hide_index=True, column_config={
    "Text match": st.column_config.ProgressColumn(
        "Text match",
        min_value=0.0,
        max_value=1.0,
        format="%.3f",
        help="TF-IDF cosine similarity between the resume text and the job description (0 to 1)."
    ),
    "Skill coverage %": st.column_config.ProgressColumn(
        "Skill coverage %",
        min_value=0,
        max_value=100,
        format="%.0f%%",
        help="Percentage of required skills from the job description that appear in the resume."
    ),
    "Verified coverage %": st.column_config.ProgressColumn(
        "Verified coverage %",
        min_value=0,
        max_value=100,
        format="%.0f%%",
        help="Percentage of required skills supported by evidence in experience or projects."
    ),
    "Trust": st.column_config.TextColumn(
        "Trust",
        help="Integrity status: OK, Review (unverified claims), or High risk (hidden text / manipulation)."
    ),
    "Notes": st.column_config.NumberColumn(
        "Notes",
        help="Number of authenticity warning signs flagged by the screening engine."
    ),
})
csv = final.copy()
csv["Matched skills"] = ["; ".join(R["skill_res"][labels.index(c)]["matched"]) for c in csv["Candidate"]]
csv["Missing skills"] = ["; ".join(R["skill_res"][labels.index(c)]["missing"]) for c in csv["Candidate"]]
csv["Authenticity notes"] = [" | ".join(f["title"] for f in integ[labels.index(c)]["flags"]) for c in csv["Candidate"]]
st.download_button("⬇ Download results as CSV", csv.to_csv(index=False).encode("utf-8"),
                   file_name="candidate_analysis.csv", mime="text/csv")
if n < 3:
    st.warning("With fewer than 3 resumes the text-match scores are only a rough guide.")

# ============================================================================= authenticity check
st.markdown("---")
st.header("Authenticity check")
st.caption("Looks for text a human cannot see (white, near-white, microscopic or off-page text) and for skills that are "
           "only *listed* but never shown in experience or projects.")
flagged = [i for i in range(n) if integ[i]["trust"] != "✅ OK"]
if not flagged:
    st.success("No hidden text and no suspicious keyword patterns were found.")
for i in flagged:
    g = integ[i]
    with st.container(border=True):
        st.markdown(f"### {g['trust']} &nbsp; {labels[i]}")
        for f in g["flags"]:
            icon = {"high": "🔴", "medium": "🟠", "low": "🟡"}[f["severity"]]
            st.markdown(f"{icon} **{f['title']}** - {f['detail']}")
        if g["hidden_text"]:
            st.markdown("**Hidden text (not visible to a human reader):**")
            st.code(g["hidden_text"][:400] + (" ..." if len(g["hidden_text"]) > 400 else ""), language=None)
            nv = g.get("naive")
            if nv:
                eff = pd.DataFrame({
                    "": ["If the hidden text had been trusted", "With the hidden text ignored (used)"],
                    "Text match": [f"{nv['similarity']:.3f}", f"{R['tfidf_scores'][i]:.3f}"],
                    "Skill coverage": [f"{nv['coverage']:.0f}%", f"{R['skill_res'][i]['percentage']:.0f}%"],
                    "Position": [f"#{nv['rank']} of {n}", f"#{int(final.index[final['Candidate'] == labels[i]][0]) + 1} of {n}"]})
                st.markdown("**Effect on the ranking:**")
                st.dataframe(eff, hide_index=True)

                chart_data = pd.DataFrame({
                    "Scenario": ["Trusted (manipulated)", "Ignored (actual)"],
                    "Text Match": [nv["similarity"], float(R["tfidf_scores"][i])],
                    "Skill Coverage %": [nv["coverage"], float(R["skill_res"][i]["percentage"])]
                }).set_index("Scenario")
                st.bar_chart(chart_data)
minor = [i for i in range(n) if integ[i]["trust"] == "✅ OK" and integ[i]["flags"]]
if minor:
    with st.expander(f"Minor notes ({len(minor)} resume(s), no action needed)"):
        for i in minor:
            for f in integ[i]["flags"]:
                st.markdown(f"- **{labels[i]}**: {f['detail']}")
with st.expander("How the authenticity check works"):
    st.markdown("""
- **Hidden text:** every page of a PDF is drawn as an image and the colour of each piece of text is compared with the
  colour behind it. Text that is the same colour as its background, in a microscopic font, invisible, or outside the
  page is treated as hidden. White text on a *dark* banner is normal and is not flagged. Hidden text is **removed
  before scoring**, so it cannot raise a candidate's rank.
- **Skill evidence:** each skill is checked for proof in Experience / Projects. Skills that appear only in the skills
  list or the summary are *unverified*. **Verified coverage** counts only required skills that are backed by evidence.
- **Other signs:** an unusually long skill list, or the same keyword repeated many times.
- **Limits:** these are warning signs for a human reviewer, not proof of dishonesty. Honest resumes may list a skill
  without a project, which is why only strong patterns raise a flag. Scanned PDFs and TXT files cannot be checked for
  hidden text.""")

# ============================================================================= candidate profile
st.markdown("---")
st.header("Candidate profile")
who = st.selectbox("Choose a candidate", range(n), format_func=lambda i: labels[i], key="profile")
x, r, g = infos[who], R["skill_res"][who], integ[who]
with st.container(border=True):
    p1, p2 = st.columns([2, 1])
    with p1:
        st.markdown(f"### {labels[who]} &nbsp; {g['trust']}")
        st.write(f"**University:** {', '.join(x['universities']) or 'Not detected'}")
        st.write(f"**Organizations:** {', '.join(x['organizations']) or 'Not detected'}")
        st.write(f"**Locations:** {', '.join(x['locations']) or 'Not detected'}")
        st.write(f"**Job titles:** {', '.join(x['job_titles']) or 'Not detected'}")
        st.markdown("**Summary**")
        st.write(R["summaries"][who])
    with p2:
        st.metric("Text match", f"{R['tfidf_scores'][who]:.3f}")
        st.metric("Skill coverage", f"{r['percentage']:.0f}%")
        st.metric("Verified coverage", f"{g['verified_coverage']:.0f}%")
    ev = g["evidence"]
    st.markdown("**Required skills**")
    chip_html = ""
    for s in R["jd_skills"]:
        if s not in x["skills"]:
            chip_html += f"<span class='chip bad'>✗ {s} - missing</span>"
        elif ev.get(s) == "demonstrated":
            chip_html += f"<span class='chip ok'>✓ {s} - shown in experience</span>"
        elif ev.get(s) == "unknown":
            chip_html += f"<span class='chip ok'>✓ {s}</span>"
        else:
            chip_html += f"<span class='chip warn'>◐ {s} - listed only</span>"
    st.markdown(chip_html or "<span class='chip gray'>No required skills detected</span>", unsafe_allow_html=True)
    st.markdown("**Other skills**")
    st.markdown(chips(r["additional"], "info"), unsafe_allow_html=True)
    if g["flags"]:
        st.markdown("**Authenticity notes**")
        for f in g["flags"]:
            st.markdown(f"- {f['title']}: {f['detail']}")

# ============================================================================= pool insights
if R["can_cluster"]:
    with st.expander("Candidate pool insights (groups and profile areas)"):
        st.markdown("**Groups of similar candidates**")
        grp = pd.DataFrame({"Candidate": labels, "Group": R["hc"],
                            "Group keywords": [R["hc_keywords"][c] for c in R["hc"]]}).sort_values("Group")
        st.dataframe(grp, hide_index=True)
        out = [labels[i] for i, c in enumerate(R["db"]) if c == -1]
        st.markdown("**Unusual profiles (do not fit any dense group):** " + (", ".join(out) if out else "none"))
        st.markdown("**Profile areas in this pool**")
        for t, words in enumerate(R["topics"]):
            who_t = [labels[i] for i in range(n) if R["dominant_topic"][i] == t]
            st.markdown(f"{chips(words[:6], 'info')} &nbsp; _{', '.join(who_t) if who_t else '-'}_", unsafe_allow_html=True)

st.info("**Interpretation:** text match shows how closely a resume's *wording* resembles the job description; skill "
        "coverage is the share of required keywords found; verified coverage counts only skills backed by experience. "
        "Use them as a screening aid - a person should review every candidate, especially flagged ones.")

# ============================================================================= technical details (optional)
if show_tech:
    st.markdown("---")
    st.header("Technical details - NLP pipeline")
    st.caption("Every intermediate step of the analysis, for evaluation and teaching purposes.")
    render_technical(R)
