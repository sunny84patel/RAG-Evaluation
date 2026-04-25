"""
evaluation/pipeline.py
LangGraph RAG evaluation pipeline — optimised for Render free tier (512 MB RAM)

LLM  : Groq (free) — llama-3.3-70b-versatile
Embed: HuggingFace Inference API (free) — no model loaded in RAM
VDB  : Qdrant in-memory

Graph nodes:
  ingest → index → generate_qa → retrieve → answer → score → report
"""

import os
import json
import asyncio
import math
from typing import Any, TypedDict

import numpy as np

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from datasets import Dataset
from ragas import evaluate
import ragas.callbacks as _ragas_callbacks
import ragas.dataset_schema as _ragas_dataset_schema
from ragas.metrics import faithfulness, context_precision, context_recall
from ragas.metrics._answer_relevance import AnswerRelevancy
from ragas.run_config import RunConfig
from ragas.utils import safe_nanmean
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from evaluation.store import evaluation_store

# RAGAS 0.4.x: parse_run_traces does root_traces[0] with no guard. Empty roots →
# IndexError in EvaluationResult.__post_init__. dataset_schema binds the function
# at import time, so patch both the module and the re-export used by EvaluationResult.
_orig_parse_run_traces = _ragas_callbacks.parse_run_traces


def _parse_run_traces_safe(traces: dict, parent_run_id: str | None = None) -> list:
    try:
        return _orig_parse_run_traces(traces, parent_run_id)
    except IndexError:
        return []


_ragas_callbacks.parse_run_traces = _parse_run_traces_safe
_ragas_dataset_schema.parse_run_traces = _parse_run_traces_safe

# HuggingFace all-MiniLM-L6-v2 → 384-dim vectors
# API call only — zero RAM for model weights
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM   = 384


# ── State ─────────────────────────────────────────────────────────────────────

class PipelineState(TypedDict):
    session_id: str
    run_id:     str
    file_path:  str
    config:     dict
    documents:  list
    chunks:     list
    collection_name:     str
    qa_pairs:            list
    retrieved_contexts:  list
    generated_answers:   list
    ragas_scores:        dict
    error:               str | None
    _qdrant_client:      Any
    _vector_store:       Any


# ── Shared factory functions ──────────────────────────────────────────────────

def _get_llm(config: dict) -> ChatGroq:
    return ChatGroq(
        model=config.get("llm_model", "llama-3.3-70b-versatile"),
        temperature=0,
        groq_api_key=os.environ["GROQ_API_KEY"],
    )


def _get_embeddings() -> HuggingFaceEndpointEmbeddings:
    """
    Uses HuggingFace Inference API — no model loaded locally.
    Free tier supports sentence-transformers models.
    Get token at: https://huggingface.co/settings/tokens
    """
    return HuggingFaceEndpointEmbeddings(
        model=EMBED_MODEL,
        huggingfacehub_api_token=os.environ["HF_API_KEY"],
    )


def _update(run_id: str, progress: int, step: str):
    evaluation_store.update_run(
        run_id, {"progress": progress, "current_step": step, "status": "running"}
    )


# RAGAS 0.4+ returns EvaluationResult; each metric is a list of per-row scores.
_RAGAS_METRIC_KEYS = (
    "answer_relevancy",
    "faithfulness",
    "context_precision",
    "context_recall",
)


def _ragas_metric_mean(result: Any, key: str) -> float:
    """Turn a RAGAS metric value (scalar or per-row list) into one float (nan if missing/empty)."""
    try:
        raw = result[key]
    except (KeyError, TypeError):
        return float("nan")
    if isinstance(raw, (list, tuple)):
        if len(raw) == 0:
            return float("nan")
        return float(safe_nanmean([float(x) for x in raw]))
    if raw is None:
        return float("nan")
    v = float(raw)
    return v if not math.isnan(v) else float("nan")


# ── Chunking ──────────────────────────────────────────────────────────────────

def _chunk(docs: list[Document], config: dict) -> list[Document]:
    strategy = config.get("chunk_strategy", "sentence_window")
    size     = config.get("chunk_size", 512)
    overlap  = config.get("chunk_overlap", 64)

    if strategy == "fixed":
        return RecursiveCharacterTextSplitter(
            chunk_size=size, chunk_overlap=overlap
        ).split_documents(docs)

    elif strategy == "parent_child":
        parents  = RecursiveCharacterTextSplitter(
            chunk_size=size * 2, chunk_overlap=overlap
        ).split_documents(docs)
        children = []
        child_sp = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=overlap)
        for i, p in enumerate(parents):
            for kid in child_sp.split_documents([p]):
                kid.metadata["parent_id"] = i
                children.append(kid)
        return children

    # sentence_window (default)
    return RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    ).split_documents(docs)


# ── Pipeline nodes ────────────────────────────────────────────────────────────

def ingest_node(state: PipelineState) -> PipelineState:
    _update(state["run_id"], 10, "Loading and chunking document...")
    try:
        loader = PyPDFLoader(state["file_path"]) \
                 if state["file_path"].endswith(".pdf") \
                 else TextLoader(state["file_path"])
        docs   = loader.load()
        chunks = _chunk(docs, state["config"])
        return {**state, "documents": docs, "chunks": chunks}
    except Exception as e:
        return {**state, "error": f"Ingest failed: {e}"}


def index_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    _update(state["run_id"], 25, "Calling HuggingFace API for embeddings → Qdrant...")
    try:
        col        = f"eval_{state['run_id'].replace('-','_')[:16]}"
        client     = QdrantClient(":memory:")
        embeddings = _get_embeddings()

        client.create_collection(
            collection_name=col,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        vs = QdrantVectorStore(client=client, collection_name=col, embedding=embeddings)
        vs.add_documents(state["chunks"])
        return {**state, "collection_name": col, "_qdrant_client": client, "_vector_store": vs}
    except Exception as e:
        return {**state, "error": f"Indexing failed: {e}"}


def generate_qa_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    _update(state["run_id"], 42, "Generating eval Q&A pairs via Groq LLM...")
    try:
        llm   = _get_llm(state["config"])
        num_q = state["config"].get("num_eval_questions", 8)
        text  = "\n\n".join(c.page_content for c in state["chunks"][:20])

        prompt = f"""Generate exactly {num_q} diverse question-answer pairs from the content below.
Test factual recall, reasoning, and multi-hop understanding.
Respond ONLY with a valid JSON array — no markdown fences:
[{{"question":"...","ground_truth":"..."}}]

Content:
{text[:5000]}"""

        raw = llm.invoke(prompt).content.strip()
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        qa_pairs = json.loads(raw.strip())
        return {**state, "qa_pairs": qa_pairs[:num_q]}
    except Exception as e:
        return {**state, "error": f"QA generation failed: {e}"}


def retrieve_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    _update(state["run_id"], 58, "Retrieving contexts from Qdrant...")
    try:
        vs = state["_vector_store"]
        k  = state["config"].get("retriever_k", 5)
        contexts = [
            [d.page_content for d in vs.similarity_search(qa["question"], k=k)]
            for qa in state["qa_pairs"]
        ]
        return {**state, "retrieved_contexts": contexts}
    except Exception as e:
        return {**state, "error": f"Retrieval failed: {e}"}


def answer_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    _update(state["run_id"], 72, "Generating answers via Groq LLM...")
    try:
        llm     = _get_llm(state["config"])
        answers = []
        for qa, ctx_list in zip(state["qa_pairs"], state["retrieved_contexts"]):
            prompt = (
                "Answer using ONLY the context below. "
                "If not found, say 'I don't know.'\n\n"
                f"Context:\n{chr(10).join(ctx_list)}\n\n"
                f"Question: {qa['question']}\nAnswer:"
            )
            answers.append(llm.invoke(prompt).content.strip())
        return {**state, "generated_answers": answers}
    except Exception as e:
        return {**state, "error": f"Answer generation failed: {e}"}


def score_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        return state
    _update(state["run_id"], 86, "Running RAGAS metrics (Groq + HF embeddings)...")
    try:
        if not state["qa_pairs"]:
            return {**state, "error": "RAGAS scoring skipped: no Q&A pairs to score."}
        n = len(state["qa_pairs"])
        if len(state["generated_answers"]) != n or len(state["retrieved_contexts"]) != n:
            return {
                **state,
                "error": "RAGAS scoring skipped: answers/contexts length mismatch with Q&A pairs.",
            }

        ragas_llm   = LangchainLLMWrapper(_get_llm(state["config"]))
        ragas_embed = LangchainEmbeddingsWrapper(_get_embeddings())

        dataset = Dataset.from_dict({
            "question":     [qa["question"]     for qa in state["qa_pairs"]],
            "answer":       state["generated_answers"],
            "contexts":     state["retrieved_contexts"],
            "ground_truth": [qa["ground_truth"] for qa in state["qa_pairs"]],
        })

        # Groq chat completions allow n<=1 only; AnswerRelevancy defaults to strictness=3 (n=3).
        metrics = [
            AnswerRelevancy(strictness=1),
            faithfulness,
            context_precision,
            context_recall,
        ]
        for m in metrics:
            m.llm        = ragas_llm
            m.embeddings = ragas_embed

        run_config = RunConfig(timeout=300, max_workers=8)
        result = evaluate(dataset, metrics=metrics, run_config=run_config)

        raw = {k: _ragas_metric_mean(result, k) for k in _RAGAS_METRIC_KEYS}
        scores: dict[str, Any] = {}
        for k, v in raw.items():
            scores[k] = None if math.isnan(v) else round(v, 4)
        finite = [raw[k] for k in _RAGAS_METRIC_KEYS if not math.isnan(raw[k])]
        if finite:
            scores["composite"] = round(float(np.mean(finite)), 4)
        else:
            scores["composite"] = None
        return {**state, "ragas_scores": scores}
    except Exception as e:
        return {**state, "error": f"RAGAS scoring failed: {e}"}


def report_node(state: PipelineState) -> PipelineState:
    if state.get("error"):
        evaluation_store.mark_failed(state["run_id"], state["error"])
        return state
    _update(state["run_id"], 98, "Saving results...")
    eval_data = [
        {"question": qa["question"], "ground_truth": qa["ground_truth"],
         "answer": ans, "contexts": ctx}
        for qa, ans, ctx in zip(
            state["qa_pairs"], state["generated_answers"], state["retrieved_contexts"]
        )
    ]
    evaluation_store.mark_complete(state["run_id"], state["ragas_scores"], eval_data)
    return state


# ── Build graph ───────────────────────────────────────────────────────────────

def _build_graph():
    g = StateGraph(PipelineState)
    for name, fn in [
        ("ingest", ingest_node), ("index", index_node),
        ("generate_qa", generate_qa_node), ("retrieve", retrieve_node),
        ("answer", answer_node), ("score", score_node), ("report", report_node),
    ]:
        g.add_node(name, fn)
    g.set_entry_point("ingest")
    for a, b in [("ingest","index"),("index","generate_qa"),("generate_qa","retrieve"),
                 ("retrieve","answer"),("answer","score"),("score","report"),("report",END)]:
        g.add_edge(a, b)
    return g.compile()


PIPELINE = _build_graph()


def _invoke_pipeline_sync(
    session_id: str,
    run_id: str,
    file_path: str,
    config: dict[str, Any],
) -> None:
    """Run LangGraph + RAGAS on a worker thread (no asyncio loop → RAGAS can use asyncio.run)."""
    PIPELINE.invoke({
        "session_id": session_id,
        "run_id": run_id,
        "file_path": file_path,
        "config": config,
        "documents": [],
        "chunks": [],
        "collection_name": "",
        "qa_pairs": [],
        "retrieved_contexts": [],
        "generated_answers": [],
        "ragas_scores": {},
        "error": None,
        "_qdrant_client": None,
        "_vector_store": None,
    })


async def run_evaluation_pipeline(session_id, run_id, file_path, config):
    try:
        await asyncio.to_thread(
            _invoke_pipeline_sync, session_id, run_id, file_path, config
        )
    except Exception as e:
        evaluation_store.mark_failed(run_id, str(e))
