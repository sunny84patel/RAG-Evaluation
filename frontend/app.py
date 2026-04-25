"""
frontend/app.py — RAG-Eval Studio Streamlit Dashboard
Run: streamlit run frontend/app.py
"""

import os
import time
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(
    page_title="RAG-Eval Studio",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _api_base() -> str:
    """FastAPI base URL: RAG_EVAL_API_BASE env, then Streamlit secrets, else local dev."""
    raw = (os.environ.get("RAG_EVAL_API_BASE") or "").strip()
    if not raw:
        try:
            raw = str(st.secrets["RAG_EVAL_API_BASE"]).strip()
        except (KeyError, FileNotFoundError, TypeError, RuntimeError, AttributeError):
            raw = ""
    if not raw:
        raw = "http://localhost:8000"
    return raw.rstrip("/")


API_BASE = _api_base()

st.markdown("""
<style>
.metric-card {
    background:#1e1e2e; border-radius:12px; padding:1.2rem;
    text-align:center; border:1px solid #313244; margin-bottom:0.5rem;
}
.score-val { font-size:2rem; font-weight:700; }
.stProgress > div > div { background-color:#cba6f7; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("session_id", None), ("run_ids", []), ("filename", "")]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🔬 RAG-Eval Studio")
    st.caption("Compare RAG configs with RAGAS metrics — powered by Groq (free)")
    st.divider()

    st.subheader("📄 Upload Document")
    uploaded = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if uploaded:
        if st.button("Upload & Start Session", use_container_width=True):
            resp = requests.post(
                f"{API_BASE}/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
            )
            if resp.status_code == 200:
                data = resp.json()
                st.session_state.session_id = data["session_id"]
                st.session_state.filename   = data["filename"]
                st.session_state.run_ids    = []
                st.success("Session ready ✓")
            else:
                st.error(f"Upload failed: {resp.text}")

    if st.session_state.session_id:
        st.info(f"📂 **{st.session_state.filename}**\n\n`{st.session_state.session_id[:16]}...`")

    st.divider()
    st.subheader("⚙️ Eval Configuration")

    chunk_strategy = st.selectbox(
        "Chunking Strategy",
        ["sentence_window", "parent_child", "fixed"],
        help="sentence_window = best for Q&A | parent_child = best for long docs | fixed = fastest",
    )
    chunk_size    = st.slider("Chunk Size (chars)", 128, 1024, 512, 64)
    chunk_overlap = st.slider("Overlap", 0, 256, 64, 16)
    retriever_k   = st.slider("Top-K Retrieval", 1, 10, 5)
    llm_model     = st.selectbox(
        "Groq LLM",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "gemma2-9b-it", "mixtral-8x7b-32768"],
        help="llama-3.3-70b = best quality | llama-3.1-8b = fastest",
    )
    num_questions = st.slider("Eval Questions", 3, 20, 8,
                              help="More questions = more accurate RAGAS scores, but slower")

    st.divider()
    disabled = not bool(st.session_state.session_id)
    if st.button("🚀 Run Evaluation", disabled=disabled,
                  use_container_width=True, type="primary"):
        payload = {
            "session_id": st.session_state.session_id,
            "config": {
                "chunk_strategy":     chunk_strategy,
                "chunk_size":         chunk_size,
                "chunk_overlap":      chunk_overlap,
                "retriever_k":        retriever_k,
                "llm_model":          llm_model,
                "num_eval_questions": num_questions,
            },
        }
        resp = requests.post(f"{API_BASE}/evaluate", json=payload)
        if resp.status_code == 200:
            run_id = resp.json()["run_id"]
            st.session_state.run_ids.append(run_id)
            st.success(f"Run started! `{run_id[:12]}...`")
        else:
            st.error(f"Failed: {resp.text}")


# ── Main panel ────────────────────────────────────────────────────────────────
st.title("📊 Evaluation Dashboard")

if not st.session_state.session_id:
    st.info("👈 Upload a document in the sidebar and click **Upload & Start Session**.")
    with st.expander("ℹ️ How it works"):
        st.markdown("""
1. **Upload** any PDF or TXT document
2. **Configure** a RAG strategy in the sidebar
3. **Run Evaluation** — the LangGraph pipeline will:
   - Chunk & embed your document (free, local)
   - Auto-generate Q&A pairs via Groq LLM
   - Retrieve contexts and generate answers
   - Score with 4 RAGAS metrics
4. **Compare** multiple configs on the leaderboard
        """)
    st.stop()


# ── Live progress for latest run ──────────────────────────────────────────────
if st.session_state.run_ids:
    latest_run_id = st.session_state.run_ids[-1]
    status_resp   = requests.get(f"{API_BASE}/status/{latest_run_id}")
    if status_resp.status_code == 200:
        run    = status_resp.json()
        status = run.get("status", "pending")

        if status in ("pending", "running"):
            st.subheader("⏳ Pipeline Running")
            progress = run.get("progress", 0)
            st.progress(progress / 100)
            st.caption(f"**Step:** {run.get('current_step', 'Initialising...')}  |  **{progress}%** complete")
            col1, col2, col3 = st.columns(3)
            col1.metric("Chunks processed", "✓" if progress > 25 else "⏳")
            col2.metric("Q&A generated",    "✓" if progress > 42 else "⏳")
            col3.metric("RAGAS scored",     "✓" if progress > 86 else "⏳")
            time.sleep(3)
            st.rerun()

        elif status == "failed":
            st.error(f"❌ Pipeline failed: {run.get('error')}")
            st.info("Common fixes: check your GROQ_API_KEY, or reduce number of eval questions.")


# ── Results ───────────────────────────────────────────────────────────────────
compare_resp = requests.get(f"{API_BASE}/compare/{st.session_state.session_id}")
if compare_resp.status_code != 200:
    st.info("No completed evaluations yet. Configure and run one in the sidebar ☝️")
    st.stop()

compare_data = compare_resp.json()
rankings     = compare_data.get("rankings", [])

if not rankings:
    st.info("First evaluation is running. This page refreshes automatically.")
    st.stop()


# ── Best config banner ────────────────────────────────────────────────────────
best = rankings[0]
st.success(
    f"🏆 **Best Config:** `{best['config']['chunk_strategy']}` chunking  |  "
    f"K={best['config']['retriever_k']}  |  "
    f"LLM: `{best['config']['llm_model']}`  |  "
    f"Composite Score: **{best['ragas_scores'].get('composite', 0):.4f}**"
)
st.divider()


# ── Score cards ───────────────────────────────────────────────────────────────
st.subheader("📈 RAGAS Scores — Best Configuration")
scores  = best["ragas_scores"]
metrics = [
    ("Answer\nRelevancy",   "answer_relevancy"),
    ("Faithfulness",        "faithfulness"),
    ("Context\nPrecision",  "context_precision"),
    ("Context\nRecall",     "context_recall"),
    ("Composite\nScore",    "composite"),
]
cols = st.columns(5)
for col, (label, key) in zip(cols, metrics):
    val   = scores.get(key, 0)
    color = "#a6e3a1" if val >= 0.8 else "#f38ba8" if val < 0.6 else "#fab387"
    col.markdown(
        f"""<div class="metric-card">
        <div style="font-size:0.75rem;color:#cdd6f4;white-space:pre-line">{label}</div>
        <div class="score-val" style="color:{color}">{val:.3f}</div>
        </div>""",
        unsafe_allow_html=True,
    )

st.divider()


# ── Radar chart (only if 2+ runs) ─────────────────────────────────────────────
if len(rankings) >= 2:
    st.subheader("🕸️ Config Comparison — Radar Chart")
    metric_keys   = ["answer_relevancy", "faithfulness", "context_precision", "context_recall"]
    metric_labels = ["Answer Relevancy", "Faithfulness", "Context Precision", "Context Recall"]
    fig = go.Figure()
    for run in rankings[:5]:
        label  = f"{run['config']['chunk_strategy']} | k={run['config']['retriever_k']}"
        values = [run["ragas_scores"].get(m, 0) for m in metric_keys] + \
                 [run["ragas_scores"].get(metric_keys[0], 0)]
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metric_labels + [metric_labels[0]],
            fill="toself",
            name=label,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True, template="plotly_dark", height=430,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.divider()


# ── Leaderboard ───────────────────────────────────────────────────────────────
st.subheader("🏅 All Runs — Leaderboard")
rows = []
for i, run in enumerate(rankings):
    rows.append({
        "Rank":             f"#{i+1}",
        "Strategy":         run["config"]["chunk_strategy"],
        "Chunk Size":       run["config"]["chunk_size"],
        "Top-K":            run["config"]["retriever_k"],
        "LLM":              run["config"]["llm_model"],
        "Ans. Relevancy":   f"{run['ragas_scores'].get('answer_relevancy', 0):.3f}",
        "Faithfulness":     f"{run['ragas_scores'].get('faithfulness', 0):.3f}",
        "Ctx Precision":    f"{run['ragas_scores'].get('context_precision', 0):.3f}",
        "Ctx Recall":       f"{run['ragas_scores'].get('context_recall', 0):.3f}",
        "⭐ Composite":     f"{run['ragas_scores'].get('composite', 0):.3f}",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Q&A deep dive ─────────────────────────────────────────────────────────────
st.divider()
st.subheader("🔍 Q&A Deep Dive — Best Config")
qa_items = best.get("eval_questions", [])
if qa_items:
    for i, item in enumerate(qa_items[:5]):
        with st.expander(f"Q{i+1}: {item['question'][:90]}..."):
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**📋 Ground Truth**")
                st.info(item["ground_truth"])
            with c2:
                st.markdown("**🤖 LLM Answer**")
                st.success(item["answer"])
            st.markdown("**📚 Retrieved Contexts (top 2)**")
            for j, ctx in enumerate(item.get("contexts", [])[:2]):
                st.caption(f"Context {j+1}: {ctx[:300]}...")
else:
    st.caption("Q&A data will appear after first completed run.")
