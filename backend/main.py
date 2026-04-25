"""
RAG-Eval Studio — FastAPI Backend
Author: Sunny Patel
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

from evaluation.pipeline import run_evaluation_pipeline
from evaluation.store import evaluation_store

# Repo root `.env` (works when cwd is project root or `backend/`)
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

app = FastAPI(title="RAG-Eval Studio API", version="1.0.0")

# Streamlit (hosted + local dev) — browsers send Origin without a trailing path
_CORS_ORIGINS = [
    "https://rag-evaluation-sdf.streamlit.app",
    "http://localhost:8501",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response Models ─────────────────────────────────────────────────

class EvalConfig(BaseModel):
    chunk_strategy: str = "sentence_window"   # sentence_window | parent_child | fixed
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "text-embedding-3-small"  # text-embedding-3-small | text-embedding-3-large
    retriever_k: int = 5
    llm_model: str = "gpt-4o-mini"
    num_eval_questions: int = 10


class EvalRequest(BaseModel):
    session_id: str
    config: EvalConfig


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "rag-eval-studio"}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Accept PDF/TXT upload, return session_id for subsequent eval calls."""
    if not file.filename.endswith((".pdf", ".txt")):
        raise HTTPException(400, "Only PDF and TXT files are supported.")

    session_id = str(uuid.uuid4())
    content = await file.read()

    # Persist raw bytes for the pipeline
    os.makedirs(f"/tmp/rag_eval/{session_id}", exist_ok=True)
    file_path = f"/tmp/rag_eval/{session_id}/{file.filename}"
    with open(file_path, "wb") as f:
        f.write(content)

    evaluation_store.create_session(session_id, file_path, file.filename)
    return {"session_id": session_id, "filename": file.filename, "size_bytes": len(content)}


@app.post("/evaluate")
async def start_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    """Kick off the LangGraph evaluation pipeline in the background."""
    session = evaluation_store.get_session(request.session_id)
    if not session:
        raise HTTPException(404, "Session not found. Please upload a document first.")

    run_id = str(uuid.uuid4())
    evaluation_store.create_run(request.session_id, run_id, request.config.model_dump())

    background_tasks.add_task(
        run_evaluation_pipeline,
        session_id=request.session_id,
        run_id=run_id,
        file_path=session["file_path"],
        config=request.config.model_dump(),
    )

    return {"run_id": run_id, "status": "started", "message": "Evaluation pipeline running..."}


@app.get("/status/{run_id}")
async def get_run_status(run_id: str):
    """Poll for pipeline progress and partial results."""
    run = evaluation_store.get_run(run_id)
    if not run:
        raise HTTPException(404, "Run not found.")
    return run


@app.get("/results/{session_id}")
async def get_all_results(session_id: str):
    """Return all completed runs for a session (for comparison dashboard)."""
    runs = evaluation_store.get_session_runs(session_id)
    if not runs:
        raise HTTPException(404, "No results found for this session.")
    return {"session_id": session_id, "runs": runs}


@app.get("/compare/{session_id}")
async def compare_configs(session_id: str):
    """
    Compare all completed eval runs and surface the best configuration
    ranked by composite RAGAS score.
    """
    runs = evaluation_store.get_session_runs(session_id)
    completed = [r for r in runs if r.get("status") == "completed"]
    if not completed:
        return {"message": "No completed runs yet.", "rankings": []}

    def composite_rank_key(run: dict) -> float:
        v = (run.get("ragas_scores") or {}).get("composite")
        if v is None:
            return float("-inf")
        return float(v)

    ranked = sorted(completed, key=composite_rank_key, reverse=True)
    best_rs = ranked[0].get("ragas_scores") or {}
    best_c = best_rs.get("composite")
    return {
        "session_id": session_id,
        "best_run_id": ranked[0]["run_id"],
        "best_config": ranked[0]["config"],
        "best_composite_score": 0 if best_c is None else best_c,
        "rankings": ranked,
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
