"""Evaluate retrieval and answering against eval/questions.json.

Run from backend/ (install requirements-eval.txt first):
    python -m eval.run_eval --user-id <uuid>
    python -m eval.run_eval --user-id <uuid> --skip-ragas --limit 5   # quick check

Retrieval (answerable questions only): Recall@5, MRR@10 and nDCG@10 for
vector, bm25, hybrid (RRF) and hybrid+rerank, with page-level relevance (see
eval/metrics.py). Each question is embedded once and queried once per index;
a mode's latency is the sum of the steps it uses (embed, vector query, BM25,
fusion, rerank), so modes are compared on the same results.

Answers: the real basic and agentic pipelines, scored with RAGAS faithfulness
and answer_relevancy (answerable questions, judge JUDGE_MODEL). Unanswerable
questions are scored on whether the pipeline declined. For every run it also
records p50/p95 latency (end to end, and time spent in retrieval), LLM calls,
tokens and errors, by wrapping the Groq client and search() both pipelines
use. Retrieval errors the app hides (search() returns []) are counted here.

Writes results.json (everything, per question) and results.md (tables), with
the config and git commit they came from.
"""

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core import config
from eval.corpus import load_user_docs
from eval.metrics import (
    QuestionError,
    decline_reason,
    finite_or_none,
    mean,
    mrr_at_k,
    ndcg_at_k,
    percentile,
    recall_at_k,
    validate_questions,
)
from services import agentic_rag, embedder, embeddings, rag

EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parents[1]
RETRIEVAL_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
PIPELINES = ("basic", "agentic")
RECALL_K = 5
RANK_K = 10
JUDGE_MODEL = "qwen/qwen3.8-27b"
# (HYBRID_SEARCH_ENABLED, RERANK_ENABLED) for --answer-retrieval.
ANSWER_RETRIEVAL_FLAGS = {
    "vector": (False, False),
    "hybrid": (True, False),
    "hybrid_rerank": (True, True),
}


# ── instrumentation ─────────────────────────────────────────────────────────

@dataclass
class RunStats:
    """What one pipeline run did. Updated from worker threads, hence the lock."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    search_calls: int = 0
    retrieval_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **amounts: float) -> None:
        with self._lock:
            for name, amount in amounts.items():
                setattr(self, name, getattr(self, name) + amount)

    def error(self, where: str, exc: BaseException) -> None:
        with self._lock:
            self.errors.append(f"{where}: {type(exc).__name__}: {exc}"[:500])


# Set while a pipeline runs; asyncio tasks and to_thread calls inherit it.
_current_run: contextvars.ContextVar[RunStats | None] = contextvars.ContextVar("eval_run", default=None)
_answer_models: set[str] = set()


def _install_instrumentation() -> None:
    """Count every Groq call and search() made by the basic and agentic pipelines."""
    completions = rag.get_groq_client().chat.completions
    original_create = completions.create

    def create(*args, **kwargs):
        stats = _current_run.get()
        if kwargs.get("model"):
            _answer_models.add(kwargs["model"])
        try:
            response = original_create(*args, **kwargs)
        except Exception as exc:
            if stats:
                stats.add(llm_calls=1)
                stats.error("llm", exc)
            raise
        if stats:
            usage = getattr(response, "usage", None)
            stats.add(
                llm_calls=1,
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        return response

    completions.create = create

    async def tracked_search(query: str, n_results: int = 5, doc_id: str | None = None, *, user_id: str):
        stats = _current_run.get()
        started = time.perf_counter()
        try:
            return await embedder.search_or_raise(query, n_results, doc_id, user_id=user_id)
        except Exception as exc:
            if stats:
                stats.error("search", exc)
            return []  # what the app's search() does
        finally:
            if stats:
                stats.add(search_calls=1, retrieval_s=time.perf_counter() - started)

    rag.search = tracked_search
    agentic_rag.search = tracked_search


def _set_answer_retrieval(mode: str) -> str:
    """Point both pipelines at one retrieval mode ("config" keeps .env's); returns the mode in use."""
    if mode != "config":
        hybrid, rerank = ANSWER_RETRIEVAL_FLAGS[mode]
        embedder.HYBRID_SEARCH_ENABLED = hybrid
        embedder.RERANK_ENABLED = rerank
        rag.RERANK_ENABLED = rerank
    flags = (embedder.HYBRID_SEARCH_ENABLED, embedder.RERANK_ENABLED)
    if flags == (False, True):
        return "vector"  # rerank only applies to hybrid results
    return {v: k for k, v in ANSWER_RETRIEVAL_FLAGS.items()}[flags]


# ── retrieval ───────────────────────────────────────────────────────────────

def _hit_refs(hits: list[dict]) -> list[list]:
    return [[h["doc_name"], h["page_number"]] for h in hits]


async def _retrieve_all_modes(
    question: dict, modes: tuple[str, ...], user_id: str, doc_id: str | None, total_chunks: int
) -> dict[str, dict]:
    query = question["question"]
    depth = min(max(RANK_K, config.RERANK_CANDIDATES), total_chunks)
    timing: dict[str, float] = {}
    vector_hits: list[dict] | None = None
    bm25_hits: list[dict] | None = None
    vector_error = bm25_error = None

    if set(modes) - {"bm25"}:
        try:
            started = time.perf_counter()
            embedding = await embedder.get_embedding(query)
            timing["embed"] = time.perf_counter() - started
            started = time.perf_counter()
            vector_hits = await asyncio.to_thread(embedder._vector_search, embedding, depth, user_id, doc_id)
            timing["vector"] = time.perf_counter() - started
        except Exception as exc:
            vector_error = f"{type(exc).__name__}: {exc}"
    if set(modes) - {"vector"}:
        try:
            started = time.perf_counter()
            bm25_hits = embedder._search_bm25(query, depth, user_id, doc_id)
            timing["bm25"] = time.perf_counter() - started
        except Exception as exc:
            bm25_error = f"{type(exc).__name__}: {exc}"

    results: dict[str, dict] = {}
    for mode in modes:
        hits: list[dict] = []
        steps: list[str] = []
        error = None
        try:
            if mode == "vector":
                error = vector_error
                hits, steps = (vector_hits or [])[:RANK_K], ["embed", "vector"]
            elif mode == "bm25":
                error = bm25_error
                hits, steps = (bm25_hits or [])[:RANK_K], ["bm25"]
            else:
                error = vector_error or bm25_error
                if not error:
                    steps = ["embed", "vector", "bm25", f"fuse_{mode}"]
                    started = time.perf_counter()
                    if mode == "hybrid":
                        hits = embedder._fuse_with_rrf(vector_hits[:RANK_K], bm25_hits[:RANK_K], RANK_K)
                        timing[f"fuse_{mode}"] = time.perf_counter() - started
                    else:
                        fused = embedder._fuse_with_rrf(vector_hits, bm25_hits, depth)
                        timing[f"fuse_{mode}"] = time.perf_counter() - started
                        started = time.perf_counter()
                        hits = await embedder._rerank_hits(query, fused, RANK_K)
                        timing["rerank"] = time.perf_counter() - started
                        steps.append("rerank")
        except Exception as exc:
            error, hits = f"{type(exc).__name__}: {exc}", []

        args = (hits, question["doc_name"], question["expected_pages"])
        results[mode] = {
            # A failed query scores 0, as the app would have retrieved nothing.
            f"recall@{RECALL_K}": recall_at_k(*args, RECALL_K),
            f"mrr@{RANK_K}": mrr_at_k(*args, RANK_K),
            f"ndcg@{RANK_K}": ndcg_at_k(*args, RANK_K),
            "latency_ms": None if error else round(sum(timing[s] for s in steps) * 1000, 1),
            "retrieved": _hit_refs(hits),
            "error": error,
        }
    return results


# ── answers ─────────────────────────────────────────────────────────────────

async def _run_pipeline(pipeline: str, question: dict, user_id: str, doc_id: str | None) -> dict:
    stats = RunStats()
    token = _current_run.set(stats)
    answer = has_answer = None
    contexts: list[dict] = []
    started = time.perf_counter()
    try:
        if pipeline == "basic":
            hits_seen: list[dict] = []
            search = rag.search

            async def capturing_search(*args, **kwargs):
                hits = await search(*args, **kwargs)
                hits_seen.extend(hits)
                return hits

            rag.search = capturing_search
            try:
                result = await rag.answer_query(question["question"], [], doc_id, user_id=user_id)
            finally:
                rag.search = search
            answer, has_answer = result["answer"], result["has_answer"]
            # What the answer model saw: the same relevance cut answer_query applies.
            contexts = hits_seen if rag.RERANK_ENABLED else [h for h in hits_seen if h["relevance_score"] > 0.1]
        else:
            state = await agentic_rag.agentic_graph.ainvoke(
                agentic_rag.initial_state(question["question"], [], user_id, doc_id)
            )
            answer, has_answer = state["final_answer"], state["has_answer"]
            _, contexts, _ = agentic_rag._build_synthesis_context(state)
    except Exception as exc:
        stats.error("pipeline", exc)
    finally:
        elapsed = time.perf_counter() - started
        _current_run.reset(token)

    reason = decline_reason(answer, has_answer) if answer is not None else None
    return {
        "answer": answer,
        "has_answer": has_answer,
        "declined": reason is not None,
        "decline_reason": reason,
        "latency_ms": round(elapsed * 1000, 1),
        "retrieval_ms": round(stats.retrieval_s * 1000, 1),
        "llm_calls": stats.llm_calls,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "search_calls": stats.search_calls,
        "errors": stats.errors,
        "contexts": _hit_refs(contexts),
        "_context_texts": [c["chunk_text"] for c in contexts],
        "faithfulness": None,
        "answer_relevancy": None,
    }


# ── RAGAS ───────────────────────────────────────────────────────────────────

def _ragas_score(rows: list[dict], workers: int, timeout: int) -> list[dict]:
    """faithfulness + answer_relevancy per row ({user_input, response, retrieved_contexts, reference})."""
    from langchain_core.embeddings import Embeddings
    from langchain_groq import ChatGroq
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._faithfulness import Faithfulness
    from ragas.run_config import RunConfig

    class GeminiEmbeddings(Embeddings):
        """The app's embedding model, for answer_relevancy's question similarity."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            result = embeddings._get_client().models.embed_content(
                model=embeddings.EMBEDDING_MODEL, contents=texts
            )
            return [list(e.values) for e in result.embeddings]

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]

        async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
            return await asyncio.to_thread(self.embed_documents, texts)

        async def aembed_query(self, text: str) -> list[float]:
            return await asyncio.to_thread(self.embed_query, text)

    judge = ChatGroq(
        model=JUDGE_MODEL,
        api_key=config.GROQ_API_KEY,
        temperature=0,
        reasoning_effort="none",
        max_retries=6,
    )
    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(), AnswerRelevancy()],
        llm=LangchainLLMWrapper(judge),
        embeddings=LangchainEmbeddingsWrapper(GeminiEmbeddings()),
        run_config=RunConfig(max_workers=workers, timeout=timeout),
        raise_exceptions=False,
    )
    return [
        {
            "faithfulness": finite_or_none(row.get("faithfulness")),
            "answer_relevancy": finite_or_none(row.get("answer_relevancy")),
        }
        for row in result.to_pandas().to_dict(orient="records")
    ]


def _score_answers(per_question: list[dict], pipelines: tuple[str, ...], workers: int, timeout: int) -> None:
    for pipeline in pipelines:
        targets = [
            q for q in per_question
            if q["type"] != "unanswerable" and q["answers"][pipeline]["answer"] is not None
        ]
        if not targets:
            continue
        print(f"\nRAGAS: scoring {len(targets)} {pipeline} answers with {JUDGE_MODEL}")
        rows = [
            {
                "user_input": q["question"],
                "response": q["answers"][pipeline]["answer"],
                "retrieved_contexts": q["answers"][pipeline]["_context_texts"],
                "reference": q["reference_answer"],
            }
            for q in targets
        ]
        for q, scores in zip(targets, _ragas_score(rows, workers, timeout)):
            q["answers"][pipeline].update(scores)


# ── summaries ───────────────────────────────────────────────────────────────

def _latency(values_ms: list[float | None]) -> dict:
    present = [v for v in values_ms if v is not None]
    return {"p50_ms": percentile(present, 50), "p95_ms": percentile(present, 95)}


def _summarize_retrieval(per_question: list[dict], modes: tuple[str, ...]) -> dict:
    summary = {}
    for mode in modes:
        runs = [q["retrieval"][mode] for q in per_question if mode in q.get("retrieval", {})]
        summary[mode] = {
            "n": len(runs),
            **{
                metric: mean(r[metric] for r in runs)
                for metric in (f"recall@{RECALL_K}", f"mrr@{RANK_K}", f"ndcg@{RANK_K}")
            },
            "latency": _latency([r["latency_ms"] for r in runs]),
            "errors": sum(r["error"] is not None for r in runs),
            # No LLM in retrieval; tokens and LLM calls are reported for answer pipelines.
        }
    return summary


def _summarize_answers(per_question: list[dict], pipelines: tuple[str, ...]) -> dict:
    summary = {}
    for pipeline in pipelines:
        runs = [(q, q["answers"][pipeline]) for q in per_question if pipeline in q.get("answers", {})]
        answerable = [r for q, r in runs if q["type"] != "unanswerable"]
        unanswerable = [r for q, r in runs if q["type"] == "unanswerable"]
        answered = [r for _, r in runs if r["answer"] is not None]
        n = len(runs)
        summary[pipeline] = {
            "n": n,
            "faithfulness": mean(r["faithfulness"] for r in answerable),
            "faithfulness_scored": sum(r["faithfulness"] is not None for r in answerable),
            "answer_relevancy": mean(r["answer_relevancy"] for r in answerable),
            "answer_relevancy_scored": sum(r["answer_relevancy"] is not None for r in answerable),
            "unanswerable": {
                "n": len(unanswerable),
                "declined": sum(r["declined"] for r in unanswerable if r["answer"] is not None),
                "decline_rate": mean(float(r["declined"]) for r in unanswerable if r["answer"] is not None),
            },
            "answerable_declined": {
                "n": len(answerable),
                "declined": sum(r["declined"] for r in answerable if r["answer"] is not None),
                "rate": mean(float(r["declined"]) for r in answerable if r["answer"] is not None),
            },
            "latency_end_to_end": _latency([r["latency_ms"] for r in answered]),
            "latency_retrieval": _latency([r["retrieval_ms"] for r in answered]),
            "llm_calls": {"total": sum(r["llm_calls"] for _, r in runs), "per_question": _per(runs, "llm_calls")},
            "prompt_tokens": {"total": sum(r["prompt_tokens"] for _, r in runs), "per_question": _per(runs, "prompt_tokens")},
            "completion_tokens": {
                "total": sum(r["completion_tokens"] for _, r in runs),
                "per_question": _per(runs, "completion_tokens"),
            },
            "errors": {
                "questions_with_errors": sum(bool(r["errors"]) for _, r in runs),
                "total": sum(len(r["errors"]) for _, r in runs),
                "failed_runs": n - len(answered),
            },
        }
    return summary


def _per(runs: list[tuple[dict, dict]], key: str) -> float | None:
    return sum(r[key] for _, r in runs) / len(runs) if runs else None


# ── provenance ──────────────────────────────────────────────────────────────

def _git_info() -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=REPO_DIR, text=True, stderr=subprocess.DEVNULL).strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_version(name: str) -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _config(args: argparse.Namespace, questions: list[dict], questions_bytes: bytes, answer_retrieval: str) -> dict:
    return {
        "questions_file": str(args.questions),
        "questions_sha256": hashlib.sha256(questions_bytes).hexdigest(),
        "questions": len(questions),
        "questions_by_type": dict(Counter(q["type"] for q in questions)),
        "limit": args.limit,
        "scope": args.scope,
        "user_id": args.user_id,
        "chroma_database": config.CHROMA_DATABASE,
        "retrieval": {
            "modes": list(args.modes),
            "recall_k": RECALL_K,
            "rank_k": RANK_K,
            "relevance": "page-level: doc_name matches and page in expected_pages",
            "embedding_model": embeddings.EMBEDDING_MODEL,
            "chunk_size": embedder.CHUNK_SIZE,
            "chunk_overlap": embedder.CHUNK_OVERLAP,
            "rrf_k": embedder._RRF_K,
            "rerank_model": config.RERANK_MODEL,
            "rerank_candidates": config.RERANK_CANDIDATES,
        },
        "answers": {
            "pipelines": list(args.pipelines),
            "retrieval_mode": answer_retrieval,
            "answer_models": sorted(_answer_models),
            "judge_model": None if args.skip_ragas else JUDGE_MODEL,
            "ragas_metrics": [] if args.skip_ragas else ["faithfulness", "answer_relevancy"],
        },
        "versions": {
            "python": sys.version.split()[0],
            **{name: _package_version(name) for name in ("ragas", "langchain-groq", "groq", "chromadb", "flashrank")},
        },
    }


# ── output ──────────────────────────────────────────────────────────────────

def _fmt(value: float | None, digits: int = 3) -> str:
    return "—" if value is None else f"{value:.{digits}f}"


def _ms(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}"


def _markdown(results: dict) -> str:
    meta, cfg = results["meta"], results["config"]
    commit = (meta["git"]["commit"] or "unknown")[:10] + (" (uncommitted changes)" if meta["git"]["dirty"] else "")
    by_type = ", ".join(f"{n} {t}" for t, n in sorted(cfg["questions_by_type"].items()))
    lines = [
        f"# DocMind eval — {meta['timestamp']}",
        "",
        f"Commit `{commit}` · {cfg['questions']} questions ({by_type}) · scope: {cfg['scope']}",
        "",
    ]

    retrieval = results["retrieval"]
    if retrieval:
        n = next(iter(retrieval.values()))["n"]
        lines += [
            f"## Retrieval ({n} answerable questions, page-level relevance)",
            "",
            f"| Mode | Recall@{RECALL_K} | MRR@{RANK_K} | nDCG@{RANK_K} | p50 ms | p95 ms | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for mode, s in retrieval.items():
            lines.append(
                f"| {mode} | {_fmt(s[f'recall@{RECALL_K}'])} | {_fmt(s[f'mrr@{RANK_K}'])} | "
                f"{_fmt(s[f'ndcg@{RANK_K}'])} | {_ms(s['latency']['p50_ms'])} | "
                f"{_ms(s['latency']['p95_ms'])} | {s['errors']} |"
            )
        lines += ["", "Latency = embedding + queries (+ fusion, rerank) per question. No LLM calls.", ""]

    answers = results["answers"]
    if answers:
        lines += [
            f"## Answers (retrieval: {cfg['answers']['retrieval_mode']}, "
            f"judge: {cfg['answers']['judge_model'] or 'RAGAS skipped'})",
            "",
            "| Pipeline | Faithfulness | Answer relevancy | Unanswerable declined | Answerable declined "
            "| p50 / p95 end-to-end ms | p50 / p95 retrieval ms | Tokens/q in / out | LLM calls/q | Errors |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for pipeline, s in answers.items():
            una, ans = s["unanswerable"], s["answerable_declined"]
            e2e, ret = s["latency_end_to_end"], s["latency_retrieval"]
            lines.append(
                f"| {pipeline} | {_fmt(s['faithfulness'])} (n={s['faithfulness_scored']}) "
                f"| {_fmt(s['answer_relevancy'])} (n={s['answer_relevancy_scored']}) "
                f"| {una['declined']}/{una['n']} | {ans['declined']}/{ans['n']} "
                f"| {_ms(e2e['p50_ms'])} / {_ms(e2e['p95_ms'])} | {_ms(ret['p50_ms'])} / {_ms(ret['p95_ms'])} "
                f"| {_fmt(s['prompt_tokens']['per_question'], 0)} / {_fmt(s['completion_tokens']['per_question'], 0)} "
                f"| {_fmt(s['llm_calls']['per_question'], 1)} "
                f"| {s['errors']['questions_with_errors']} q ({s['errors']['total']}) |"
            )
        lines += [
            "",
            "Declines are detected from has_answer=false or refusal phrasing (see decline_reason "
            "per question in results.json). Retrieval time is summed over a run's search calls; "
            "agentic searches run concurrently, so it can exceed wall-clock time.",
            "",
        ]

    lines += ["## Config", "", "```json", json.dumps(cfg, indent=2), "```", ""]
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

async def _collect(args: argparse.Namespace, questions: list[dict]) -> list[dict]:
    docs = load_user_docs(args.user_id)
    missing = sorted({q["doc_name"] for q in questions} - set(docs))
    if missing:
        sys.exit(f"documents in the questions file that this user hasn't indexed: {', '.join(missing)}")
    for q in questions:
        absent = [p for p in q["expected_pages"] if p not in docs[q["doc_name"]].chunks]
        if absent:
            print(f"! {q['question'][:60]!r}: no indexed text on expected pages {absent}")

    total_chunks = await asyncio.to_thread(embedder.collection.count)
    if args.modes:
        # Build the BM25 index and load the reranker now, outside the timings.
        await asyncio.to_thread(embedder._rebuild_bm25_index)
        if "hybrid_rerank" in args.modes:
            embedder._validate_rerank_model(config.RERANK_MODEL)
            await asyncio.to_thread(embedder._get_reranker)

    per_question: list[dict] = []
    for index, question in enumerate(questions, start=1):
        doc_id = docs[question["doc_name"]].doc_id if args.scope == "doc" else None
        record = {**question, "retrieval": {}, "answers": {}}
        print(f"[{index}/{len(questions)}] {question['type']}: {question['question'][:80]}")
        if args.modes and question["type"] != "unanswerable":
            record["retrieval"] = await _retrieve_all_modes(question, args.modes, args.user_id, doc_id, total_chunks)
        for pipeline in args.pipelines:
            record["answers"][pipeline] = await _run_pipeline(pipeline, question, args.user_id, doc_id)
            if args.delay:
                await asyncio.sleep(args.delay)
        per_question.append(record)
    return per_question


def _modes(value: str) -> tuple[str, ...]:
    chosen = tuple(v for v in value.split(",") if v)
    bad = [v for v in chosen if v not in RETRIEVAL_MODES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown mode(s) {bad}; choose from {', '.join(RETRIEVAL_MODES)}")
    return chosen


def _pipelines(value: str) -> tuple[str, ...]:
    chosen = tuple(v for v in value.split(",") if v)
    bad = [v for v in chosen if v not in PIPELINES]
    if bad:
        raise argparse.ArgumentTypeError(f"unknown pipeline(s) {bad}; choose from {', '.join(PIPELINES)}")
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--user-id", default=os.environ.get("RAG_EVAL_USER_ID"))
    parser.add_argument("--questions", type=Path, default=EVAL_DIR / "questions.json")
    parser.add_argument("--out-dir", type=Path, default=EVAL_DIR)
    parser.add_argument(
        "--scope", choices=("all", "doc"), default="all",
        help="search all the user's documents (the app's default) or only the question's document",
    )
    parser.add_argument("--modes", type=_modes, default=RETRIEVAL_MODES, help="comma-separated; '' to skip retrieval")
    parser.add_argument("--pipelines", type=_pipelines, default=PIPELINES, help="comma-separated; '' to skip answers")
    parser.add_argument(
        "--answer-retrieval", choices=("config", *ANSWER_RETRIEVAL_FLAGS), default="config",
        help="retrieval the answer pipelines use; 'config' = HYBRID_SEARCH_ENABLED/RERANK_ENABLED from .env",
    )
    parser.add_argument("--limit", type=int, help="only the first N questions")
    parser.add_argument("--skip-ragas", action="store_true", help="no judge calls; faithfulness/relevancy stay empty")
    parser.add_argument("--judge-workers", type=int, default=2, help="concurrent RAGAS judge calls (Groq rate limits)")
    parser.add_argument("--judge-timeout", type=int, default=180)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait after each pipeline run")
    args = parser.parse_args()
    if not args.user_id:
        parser.error("provide --user-id or set RAG_EVAL_USER_ID")

    questions_bytes = args.questions.read_bytes()
    try:
        questions = validate_questions(json.loads(questions_bytes))
    except (QuestionError, json.JSONDecodeError) as exc:
        sys.exit(f"{args.questions}: {exc}")
    if not questions:
        sys.exit(f"{args.questions} has no questions; draft some with `python -m eval.generate_questions`")
    if args.limit:
        questions = questions[: args.limit]

    _install_instrumentation()
    answer_retrieval = _set_answer_retrieval(args.answer_retrieval)
    started = datetime.now(timezone.utc)
    per_question = asyncio.run(_collect(args, questions))
    # RAGAS runs its own event loop, so it goes after (not inside) asyncio.run.
    if args.pipelines and not args.skip_ragas:
        _score_answers(per_question, args.pipelines, args.judge_workers, args.judge_timeout)

    for record in per_question:
        for run in record["answers"].values():
            run.pop("_context_texts")
    results = {
        "meta": {
            "timestamp": started.isoformat(timespec="seconds"),
            "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
            "git": _git_info(),
        },
        "config": _config(args, questions, questions_bytes, answer_retrieval),
        "retrieval": _summarize_retrieval(per_question, args.modes),
        "answers": _summarize_answers(per_question, args.pipelines),
        "per_question": per_question,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = _markdown(results)
    (args.out_dir / "results.md").write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Saved {args.out_dir / 'results.json'} and {args.out_dir / 'results.md'}")


if __name__ == "__main__":
    main()
