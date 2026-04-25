"""
evaluation/store.py
Lightweight in-memory store for sessions and evaluation runs.
In production, swap with Redis or PostgreSQL.
"""

from datetime import datetime
from typing import Any


class EvaluationStore:
    def __init__(self):
        self._sessions: dict[str, dict] = {}
        self._runs: dict[str, dict] = {}

    # ── Sessions ──────────────────────────────────────────────────────────────

    def create_session(self, session_id: str, file_path: str, filename: str) -> dict:
        self._sessions[session_id] = {
            "session_id": session_id,
            "file_path": file_path,
            "filename": filename,
            "created_at": datetime.utcnow().isoformat(),
            "run_ids": [],
        }
        return self._sessions[session_id]

    def get_session(self, session_id: str) -> dict | None:
        return self._sessions.get(session_id)

    # ── Runs ──────────────────────────────────────────────────────────────────

    def create_run(self, session_id: str, run_id: str, config: dict) -> dict:
        run = {
            "run_id": run_id,
            "session_id": session_id,
            "config": config,
            "status": "pending",
            "progress": 0,
            "current_step": "Initialising pipeline...",
            "ragas_scores": {},
            "eval_questions": [],
            "retrieved_contexts": [],
            "started_at": datetime.utcnow().isoformat(),
            "completed_at": None,
            "error": None,
        }
        self._runs[run_id] = run
        if session_id in self._sessions:
            self._sessions[session_id]["run_ids"].append(run_id)
        return run

    def get_run(self, run_id: str) -> dict | None:
        return self._runs.get(run_id)

    def update_run(self, run_id: str, updates: dict) -> None:
        if run_id in self._runs:
            self._runs[run_id].update(updates)

    def get_session_runs(self, session_id: str) -> list[dict]:
        session = self._sessions.get(session_id)
        if not session:
            return []
        return [
            self._runs[rid]
            for rid in session["run_ids"]
            if rid in self._runs
        ]

    def mark_complete(self, run_id: str, ragas_scores: dict, eval_data: list) -> None:
        self.update_run(run_id, {
            "status": "completed",
            "progress": 100,
            "current_step": "Evaluation complete",
            "ragas_scores": ragas_scores,
            "eval_questions": eval_data,
            "completed_at": datetime.utcnow().isoformat(),
        })

    def mark_failed(self, run_id: str, error: str) -> None:
        self.update_run(run_id, {
            "status": "failed",
            "error": error,
            "completed_at": datetime.utcnow().isoformat(),
        })


# Singleton
evaluation_store = EvaluationStore()
