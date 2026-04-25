# 🔬 RAG-Eval Studio

> **Auto-evaluating, self-optimising RAG pipeline benchmarking tool** — upload any document corpus, compare chunking strategies and embedding models, and surface the optimal configuration using RAGAS metrics — all orchestrated by a LangGraph state machine.

[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-green)](https://langchain-ai.github.io/langgraph/)
[![RAGAS](https://img.shields.io/badge/RAGAS-0.2+-orange)](https://docs.ragas.io)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-teal)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.39+-red)](https://streamlit.io)

---

## 🎯 What It Does

Most RAG systems are deployed and never benchmarked — you don't know if `sentence_window` outperforms `parent_child` chunking for *your* specific corpus. RAG-Eval Studio solves this:

1. **Upload** any PDF or TXT document corpus
2. **Configure** chunking strategy, embedding model, retriever K, and LLM judge
3. **Run** — a 7-node LangGraph pipeline ingests, indexes, auto-generates Q&A pairs, retrieves, answers, and scores
4. **Compare** multiple configurations side-by-side via a live Streamlit dashboard with radar charts and ranked leaderboards
5. **Export** the best configuration for production use

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LangGraph Pipeline                           │
│                                                                 │
│  [Upload] → [ingest] → [index] → [generate_qa] → [retrieve]   │
│                                                        ↓        │
│                          [report] ← [score] ← [answer]         │
└─────────────────────────────────────────────────────────────────┘
        ↓                    ↓                    ↓
   Qdrant (in-mem)      RAGAS Metrics        LangSmith Traces
   Vector Storage       4 eval dimensions    Full observability
```

**Pipeline Nodes:**

| Node | Responsibility |
|------|---------------|
| `ingest` | Load PDF/TXT, apply chosen chunking strategy |
| `index` | Embed chunks with OpenAI, upsert to Qdrant |
| `generate_qa` | LLM synthesises eval questions + ground truths from corpus |
| `retrieve` | Fetch top-K contexts per question from Qdrant |
| `answer` | LLM generates answers conditioned on retrieved contexts |
| `score` | RAGAS evaluates: answer relevancy, faithfulness, context precision, context recall |
| `report` | Persist results, compute composite score, update run status |

**RAGAS Metrics Evaluated:**
- **Answer Relevancy** — Is the answer on-topic with the question?
- **Faithfulness** — Is the answer grounded in the retrieved contexts (no hallucination)?
- **Context Precision** — Are retrieved chunks actually relevant?
- **Context Recall** — Does retrieval cover all ground-truth information?
- **Composite Score** — Weighted average (primary ranking metric)

---

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/sunny84patel/rag-eval-studio
cd rag-eval-studio
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="sk-..."
export LANGCHAIN_API_KEY="ls__..."   # optional, for LangSmith tracing
export LANGCHAIN_TRACING_V2="true"   # optional
export LANGCHAIN_PROJECT="rag-eval-studio"
```

### 3. Start Backend

```bash
uvicorn backend.main:app --reload --port 8000
```

### 4. Start Frontend

```bash
streamlit run frontend/app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 🐳 Docker

```bash
docker compose up --build
```

```yaml
# docker-compose.yml
version: "3.9"
services:
  backend:
    build: .
    command: uvicorn backend.main:app --host 0.0.0.0 --port 8000
    ports: ["8000:8000"]
    env_file: .env

  frontend:
    build: .
    command: streamlit run frontend/app.py --server.port 8501
    ports: ["8501:8501"]
    depends_on: [backend]
    env_file: .env
```

---

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/upload` | Upload document, returns `session_id` |
| `POST` | `/evaluate` | Start eval pipeline with config, returns `run_id` |
| `GET` | `/status/{run_id}` | Poll pipeline progress (0–100%) |
| `GET` | `/results/{session_id}` | All completed runs for session |
| `GET` | `/compare/{session_id}` | Ranked comparison of all configs |

---

## 🧪 Chunking Strategies

| Strategy | Best For |
|----------|----------|
| `sentence_window` | Conversational Q&A, high context precision |
| `parent_child` | Long documents, multi-hop reasoning |
| `fixed` | Structured data, tables, code |

---

## 📈 Sample Results

Running on a 50-page HR Policy PDF across 3 configurations:

| Strategy | Embedding | K | Faithfulness | Answer Relevancy | Composite |
|----------|-----------|---|-------------|-----------------|-----------|
| sentence_window | text-embedding-3-small | 5 | 0.94 | 0.91 | **0.923** |
| parent_child | text-embedding-3-large | 3 | 0.89 | 0.93 | 0.901 |
| fixed | text-embedding-3-small | 7 | 0.82 | 0.87 | 0.845 |

---

## 🛠️ Tech Stack

- **Orchestration:** LangGraph (state machine with 7 nodes)
- **LLMs:** GPT-4o / GPT-4o-mini via OpenAI
- **Embeddings:** OpenAI text-embedding-3-small / large
- **Vector Store:** Qdrant (in-memory for dev, persistent for prod)
- **Evaluation:** RAGAS (Answer Relevancy, Faithfulness, Context Precision, Context Recall)
- **Observability:** LangSmith (per-node tracing, token usage, latency)
- **Backend:** FastAPI + Pydantic v2 + BackgroundTasks
- **Frontend:** Streamlit + Plotly (radar charts, leaderboards)
- **Deployment:** Docker + Render / Railway

---

## 📝 License

MIT

<img width="2870" height="1426" alt="image" src="https://github.com/user-attachments/assets/45571026-80c8-463a-9b15-cb3be73874af" />

