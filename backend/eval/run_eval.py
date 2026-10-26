"""Evaluate retrieval and answering against eval/questions.json.

Run from backend/ (install requirements-eval.txt first):
    python -m eval.run_eval --user-id <uuid>
    python -m eval.run_eval --user-id <uuid> --skip-ragas --limit 5   # quick check
    python -m eval.run_eval --user-id <uuid> --retrieval-only          # no Groq calls
    python -m eval.run_eval --user-id <uuid> --offset 23               # resume at question 24
    python -m eval.run_eval --score-only eval/results.json             # RAGAS only, on saved answers
    python -m eval.run_eval --merge eval/results.json eval/results.offset19.json

Retrieval (answerable questions only): Recall@5, MRR@10 and nDCG@10 for
vector, bm25, hybrid (RRF) and hybrid+rerank, with page-level relevance (see
eval/metrics.py). Each question is embedded once and queried once per index;
a mode's latency is the sum of the steps it uses (embed, vector query, BM25,
fusion, rerank), so modes are compared on the same results.

Answers: the real basic and agentic pipelines, scored with RAGAS faithfulness
and answer_relevancy (answerable questions, judge --judge-model). Unanswerable
questions are scored on whether the pipeline declined. For every run it also
records p50/p95 latency (end to end, and time spent in retrieval), LLM calls,
tokens and errors, by wrapping the Groq client and search() both pipelines
use. A run the app ends with an outage notice (search or Groq unavailable,
the agentic deadline) counts as a failed run with an error, not a decline.
The app never retries Groq; the eval lets the SDK retry 429s and 5xx
(EVAL_GROQ_MAX_RETRIES) so per-minute limits don't fail runs, and records
those retries and the time spent waiting on them; end-to-end latency is
reported with and without that wait. Use --sleep to space questions out when
the rate limit is the bottleneck.

Writes results.json (everything, per question) and results.md (tables), with
the config and git commit they came from. With --offset N the files are
results.offsetN.json/.md, so a resumed run never overwrites the partial
results of the run it continues.

When Groq's daily quota (tokens or requests per day) runs out, nothing else
will succeed until it resets, so the run stops: questions finished so far are
saved (the one in progress is dropped, not recorded as failed) and the
message says which --offset resumes it. Groq's quotas are per model, so the
finished answers are still judged with RAGAS unless the judge is the answer
model. --retrieval-only needs no Groq at all.

--score-only RESULTS_JSON loads a saved run and only judges it: answers that
have no scores yet (all of them with --rescore), then rewrites that run's
.json and .md. Run it after the judge's own daily limit stopped judging.
Results saved before context texts were kept can't be judged as-is;
--approx-contexts judges them against the whole cited pages, rebuilt from
the index, and says so in the report.

--merge A.json B.json [...] combines partial runs of the same questions and
config (e.g. a run and its --offset resume) into results.merged.json/.md,
recomputing every average and percentile from the per-question records.
"""

import argparse
import asyncio
import contextvars
import hashlib
import json
import os
import re
import subprocess
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import groq

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
from services.errors import SearchUnavailable

EVAL_DIR = Path(__file__).resolve().parent
REPO_DIR = EVAL_DIR.parents[1]
RETRIEVAL_MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")
PIPELINES = ("basic", "agentic")
RECALL_K = 5
RANK_K = 10
# The app's Groq client never retries; the eval retries 429s/5xx so per-minute limits don't fail runs.
EVAL_GROQ_MAX_RETRIES = 2
DEFAULT_JUDGE_MODEL = "qwen/qwen3.8-27b"
# answer_relevancy asks the judge for `strictness` generations in one request
# (n=strictness); Groq rejects n > 1 with a 400, so one generation it is.
ANSWER_RELEVANCY_STRICTNESS = 1
# (HYBRID_SEARCH_ENABLED, RERANK_ENABLED) for --answer-retrieval.
ANSWER_RETRIEVAL_FLAGS = {
    "vector": (False, False),
    "hybrid": (True, False),
    "hybrid_rerank": (True, True),
}


# ── Groq daily limit ────────────────────────────────────────────────────────

_DAILY_LIMIT = re.compile(r"\b(?:tokens|requests) per day\b|\((?:TPD|RPD)\)", re.IGNORECASE)
# e.g. "Please try again in 7m12.5s." -> "7m12.5s" (not the sentence's full stop)
_RETRY_IN = re.compile(r"try again in ((?:\d+(?:\.\d+)?[hms])+)", re.IGNORECASE)


class DailyLimitReached(BaseException):
    """Groq's per-day quota is used up; nothing more will succeed today, so the run stops.

    A BaseException, not an Exception, so the pipelines' and RAGAS's
    ``except Exception`` handlers can't turn it into an ordinary failed
    answer or a NaN score: it always reaches the run loop.
    """

    def __init__(self, cause: groq.RateLimitError) -> None:
        message = str(cause)
        retry_in = _RETRY_IN.search(message)
        self.retry_in = retry_in.group(1) if retry_in else None
        self.detail = message[:500]
        super().__init__(self.detail)


def _raise_if_daily_limit(exc: BaseException) -> None:
    if isinstance(exc, groq.RateLimitError) and _DAILY_LIMIT.search(str(exc)):
        raise DailyLimitReached(exc) from exc


# ── instrumentation ─────────────────────────────────────────────────────────

@dataclass
class RunStats:
    """What one pipeline run did. Updated from worker threads, hence the lock."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    search_calls: int = 0
    retrieval_s: float = 0.0
    rate_limit_retries: int = 0
    errors: list[str] = field(default_factory=list)
    # (start, end) perf_counter times the Groq client slept before retrying a 429.
    _rate_limit_waits: list[tuple[float, float]] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, **amounts: float) -> None:
        with self._lock:
            for name, amount in amounts.items():
                setattr(self, name, getattr(self, name) + amount)

    def error(self, where: str, exc: BaseException) -> None:
        with self._lock:
            self.errors.append(f"{where}: {type(exc).__name__}: {exc}"[:500])

    def rate_limited(self, started: float, ended: float) -> None:
        with self._lock:
            self.rate_limit_retries += 1
            self._rate_limit_waits.append((started, ended))

    @property
    def rate_limit_wait_s(self) -> float:
        """Wall-clock time with at least one call waiting on a 429.

        Concurrent waits (agentic sub-questions) overlap, so they are merged,
        not summed: the result never exceeds the run's own duration.
        """
        total, covered_until = 0.0, float("-inf")
        for started, ended in sorted(self._rate_limit_waits):
            started = max(started, covered_until)
            if ended > started:
                total += ended - started
                covered_until = ended
        return total


# Set while a pipeline runs; asyncio tasks and to_thread calls inherit it.
_current_run: contextvars.ContextVar[RunStats | None] = contextvars.ContextVar("eval_run", default=None)
_answer_models: set[str] = set()


def _install_instrumentation() -> None:
    """Count every Groq call, 429 retry and search() made by the basic and agentic pipelines."""
    client = rag.get_groq_client()
    # The Groq SDK retries 429s itself (client.max_retries times), sleeping in
    # _sleep_for_retry first, so the wrapper around create() can't see them.
    original_sleep_for_retry = client._sleep_for_retry

    def sleep_for_retry(*, response=None, **kwargs):
        stats = _current_run.get()
        if stats is None or response is None or response.status_code != 429:
            return original_sleep_for_retry(response=response, **kwargs)
        started = time.perf_counter()
        try:
            return original_sleep_for_retry(response=response, **kwargs)
        finally:
            stats.rate_limited(started, time.perf_counter())

    client._sleep_for_retry = sleep_for_retry
    client.max_retries = EVAL_GROQ_MAX_RETRIES

    completions = client.chat.completions
    original_create = completions.create

    def create(*args, **kwargs):
        stats = _current_run.get()
        if kwargs.get("model"):
            _answer_models.add(kwargs["model"])
        try:
            response = original_create(*args, **kwargs)
        except Exception as exc:
            _raise_if_daily_limit(exc)
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
            raise SearchUnavailable() from exc  # what the app's search() does
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
            if result.get("error"):
                raise RuntimeError(f"outage notice instead of an answer: {result['error']}")
            answer, has_answer = result["answer"], result["has_answer"]
            # What the answer model saw: the same relevance cut answer_query applies.
            contexts = hits_seen if rag.RERANK_ENABLED else [h for h in hits_seen if h["relevance_score"] > 0.1]
        else:
            state = await agentic_rag.agentic_graph.ainvoke(
                agentic_rag.initial_state(question["question"], [], user_id, doc_id),
                config=agentic_rag.RUN_CONFIG,
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
        "latency_excl_rate_limit_ms": round((elapsed - stats.rate_limit_wait_s) * 1000, 1),
        "retrieval_ms": round(stats.retrieval_s * 1000, 1),
        "rate_limit_retries": stats.rate_limit_retries,
        "rate_limit_wait_ms": round(stats.rate_limit_wait_s * 1000, 1),
        "llm_calls": stats.llm_calls,
        "prompt_tokens": stats.prompt_tokens,
        "completion_tokens": stats.completion_tokens,
        "search_calls": stats.search_calls,
        "errors": stats.errors,
        "contexts": _hit_refs(contexts),
        # Saved so --score-only can judge the answer later against what the model saw.
        "context_texts": [c["chunk_text"] for c in contexts],
        "faithfulness": None,
        "answer_relevancy": None,
    }


# ── RAGAS ───────────────────────────────────────────────────────────────────

def _stop_judge_on_daily_limit(judge) -> None:
    """Make a daily-limit 429 on the judge's Groq clients raise DailyLimitReached.

    RAGAS would otherwise log it and score the row NaN, for every remaining row.
    """
    sync_create, async_create = judge.client.create, judge.async_client.create

    def create(*args, **kwargs):
        try:
            return sync_create(*args, **kwargs)
        except groq.RateLimitError as exc:
            _raise_if_daily_limit(exc)
            raise

    async def acreate(*args, **kwargs):
        try:
            return await async_create(*args, **kwargs)
        except groq.RateLimitError as exc:
            _raise_if_daily_limit(exc)
            raise

    judge.client.create = create
    judge.async_client.create = acreate


def _ragas_score(rows: list[dict], judge_model: str, workers: int, timeout: int) -> list[dict]:
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
        model=judge_model,
        api_key=config.GROQ_API_KEY,
        temperature=0,
        reasoning_effort="none",
        max_retries=6,
    )
    _stop_judge_on_daily_limit(judge)
    result = evaluate(
        EvaluationDataset.from_list(rows),
        metrics=[Faithfulness(), AnswerRelevancy(strictness=ANSWER_RELEVANCY_STRICTNESS)],
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


def _is_scored(run: dict) -> bool:
    return run["faithfulness"] is not None and run["answer_relevancy"] is not None


def _score_targets(per_question: list[dict], pipeline: str, only_unscored: bool) -> list[dict]:
    """Questions whose ``pipeline`` answer RAGAS should judge."""
    return [
        q for q in per_question
        if q["type"] != "unanswerable"
        and pipeline in q["answers"]
        and q["answers"][pipeline]["answer"] is not None
        and not (only_unscored and _is_scored(q["answers"][pipeline]))
    ]


def _score_answers(
    per_question: list[dict], pipelines: tuple[str, ...], judge_model: str, workers: int, timeout: int,
    only_unscored: bool = False,
) -> None:
    """Fill in RAGAS scores, one pipeline at a time. Raises DailyLimitReached mid-way if
    the judge's quota runs out; pipelines scored before that keep their scores."""
    for pipeline in pipelines:
        targets = _score_targets(per_question, pipeline, only_unscored)
        if not targets:
            continue
        print(f"\nRAGAS: scoring {len(targets)} {pipeline} answers with {judge_model}")
        rows = [
            {
                "user_input": q["question"],
                "response": q["answers"][pipeline]["answer"],
                "retrieved_contexts": q["answers"][pipeline]["context_texts"],
                "reference": q["reference_answer"],
            }
            for q in targets
        ]
        for q, scores in zip(targets, _ragas_score(rows, judge_model, workers, timeout)):
            q["answers"][pipeline].update(scores)


def _judge(
    per_question: list[dict], pipelines: tuple[str, ...], args: argparse.Namespace, only_unscored: bool = False
) -> dict | None:
    """Run RAGAS over the answers; returns why judging stopped early, or None if it finished."""
    try:
        _score_answers(
            per_question, pipelines, args.judge_model, args.judge_workers, args.judge_timeout, only_unscored
        )
    except DailyLimitReached as limit:
        stopped = {
            "stage": "judging",
            "reason": "groq_daily_limit",
            "retry_in": limit.retry_in,
            "detail": limit.detail,
        }
        print(f"\nSTOPPED: {_stop_summary(stopped)}\n  {limit.detail}")
        return stopped
    return None


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
            "latency_end_to_end_excl_rate_limit": _latency([r["latency_excl_rate_limit_ms"] for r in answered]),
            # Retrieval makes no Groq calls, so it has no rate-limit wait to exclude.
            "latency_retrieval": _latency([r["retrieval_ms"] for r in answered]),
            "rate_limit": {
                "retries": sum(r["rate_limit_retries"] for _, r in runs),
                "questions_with_retries": sum(r["rate_limit_retries"] > 0 for _, r in runs),
                "wait_total_ms": round(sum(r["rate_limit_wait_ms"] for _, r in runs), 1),
            },
            "llm_calls":{"total": sum(r["llm_calls"] for _, r in runs), "per_question": _per(runs, "llm_calls")},
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
        "offset": args.offset,
        "limit": args.limit,
        "retrieval_only": args.retrieval_only,
        "scope": args.scope,
        "pacing": {
            "sleep_between_questions_s": args.sleep,
            "delay_after_pipeline_s": args.delay,
            "groq_max_retries": rag.get_groq_client().max_retries,
        },
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
            "judge_model": None if args.skip_ragas else args.judge_model,
            "ragas_metrics": [] if args.skip_ragas else ["faithfulness", "answer_relevancy"],
            "answer_relevancy_strictness": None if args.skip_ragas else ANSWER_RELEVANCY_STRICTNESS,
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


def _stop_summary(stopped: dict) -> str:
    """One line on why a run stopped early and how to continue it."""
    reset = f"; Groq says try again in {stopped['retry_in']}" if stopped.get("retry_in") else ""
    if stopped["stage"] == "answers":
        return (
            f"Groq daily limit reached at question {stopped['at_question']}{reset}. "
            f"{stopped['completed']} question(s) completed and saved; "
            f"resume with --offset {stopped['resume_offset']}."
        )
    return (
        f"Groq daily limit reached while judging with RAGAS{reset}. All answers were saved; "
        f"answers not yet judged have no faithfulness/answer_relevancy scores "
        f"(judge them later with --score-only on this results .json, or with another --judge-model)."
    )


def _markdown(results: dict) -> str:
    meta, cfg = results["meta"], results["config"]
    commit = (meta["git"]["commit"] or "unknown")[:10] + (" (uncommitted changes)" if meta["git"]["dirty"] else "")
    by_type = ", ".join(f"{n} {t}" for t, n in sorted(cfg["questions_by_type"].items()))
    first = cfg.get("offset", 0) + 1
    span = f"questions {first}–{first + cfg['questions'] - 1} of the file" if first > 1 else f"{cfg['questions']} questions"
    lines = [
        f"# DocMind eval — {meta['timestamp']}",
        "",
        f"Commit `{commit}` · {span} ({by_type}) · scope: {cfg['scope']}",
        "",
    ]
    stopped = meta.get("stopped")
    if stopped:
        lines += [f"> **Partial run — {_stop_summary(stopped)}**", ""]
    judging_stopped = meta.get("judging_stopped")
    if judging_stopped:
        lines += [f"> **Scores incomplete — {_stop_summary(judging_stopped)}**", ""]
    merged_from = meta.get("merged_from")
    if merged_from:
        parts = [f"`{m['file']}` (questions {m['first']}–{m['last']}, {m['timestamp']})" for m in merged_from]
        lines += [f"Merged from {len(merged_from)} runs: {'; '.join(parts)}.", ""]

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
            "| Pipeline | Faithfulness | Answer relevancy | Unanswerable declined | Answerable declined | Errors |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for pipeline, s in answers.items():
            una, ans = s["unanswerable"], s["answerable_declined"]
            lines.append(
                f"| {pipeline} | {_fmt(s['faithfulness'])} (n={s['faithfulness_scored']}) "
                f"| {_fmt(s['answer_relevancy'])} (n={s['answer_relevancy_scored']}) "
                f"| {una['declined']}/{una['n']} | {ans['declined']}/{ans['n']} "
                f"| {s['errors']['questions_with_errors']} q ({s['errors']['total']}) |"
            )
        lines += [
            "",
            "### Latency and cost",
            "",
            "| Pipeline | p50 / p95 end-to-end ms | … excl. 429 wait | p50 / p95 retrieval ms "
            "| 429 retries (questions) | 429 wait total s | Tokens/q in / out | LLM calls/q |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for pipeline, s in answers.items():
            e2e, net, ret = s["latency_end_to_end"], s["latency_end_to_end_excl_rate_limit"], s["latency_retrieval"]
            limit = s["rate_limit"]
            lines.append(
                f"| {pipeline} | {_ms(e2e['p50_ms'])} / {_ms(e2e['p95_ms'])} "
                f"| {_ms(net['p50_ms'])} / {_ms(net['p95_ms'])} "
                f"| {_ms(ret['p50_ms'])} / {_ms(ret['p95_ms'])} "
                f"| {limit['retries']} ({limit['questions_with_retries']}) "
                f"| {_fmt(limit['wait_total_ms'] / 1000, 1)} "
                f"| {_fmt(s['prompt_tokens']['per_question'], 0)} / {_fmt(s['completion_tokens']['per_question'], 0)} "
                f"| {_fmt(s['llm_calls']['per_question'], 1)} |"
            )
        lines += [
            "",
            "Declines are detected from has_answer=false or refusal phrasing (see decline_reason "
            "per question in results.json). Retrieval time is summed over a run's search calls; "
            "agentic searches run concurrently, so it can exceed wall-clock time. 429 wait is the "
            "wall-clock time the Groq client spent sleeping before retrying rate-limited calls "
            "(overlapping waits counted once); \"excl. 429 wait\" subtracts it from each run.",
            "",
        ]
        approximate = sum(
            bool(run.get("context_texts_approximate"))
            for q in results.get("per_question", []) for run in q["answers"].values()
        )
        if approximate:
            lines += [
                f"> {approximate} answer(s) were judged against the whole cited pages, rebuilt from the "
                "index (--approx-contexts), not the exact chunks the model saw; their faithfulness "
                "may be overstated.",
                "",
            ]

    lines += ["## Config", "", "```json", json.dumps(cfg, indent=2), "```", ""]
    return "\n".join(lines)


# ── main ────────────────────────────────────────────────────────────────────

async def _collect(args: argparse.Namespace, questions: list[dict]) -> tuple[list[dict], dict | None]:
    """Run every question; returns the finished records and, if the run stopped early, why."""
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
    last = args.offset + len(questions)
    for index, question in enumerate(questions, start=1):
        number = args.offset + index  # position in the questions file, 1-based
        doc_id = docs[question["doc_name"]].doc_id if args.scope == "doc" else None
        record = {"number": number, **question, "retrieval": {}, "answers": {}}
        print(f"[{number}/{last}] {question['type']}: {question['question'][:80]}")
        try:
            if args.modes and question["type"] != "unanswerable":
                record["retrieval"] = await _retrieve_all_modes(
                    question, args.modes, args.user_id, doc_id, total_chunks
                )
            for pipeline in args.pipelines:
                record["answers"][pipeline] = await _run_pipeline(pipeline, question, args.user_id, doc_id)
                if args.delay:
                    await asyncio.sleep(args.delay)
        except DailyLimitReached as limit:
            # This question is incomplete; drop it so it isn't scored as a failure.
            stopped = {
                "stage": "answers",
                "reason": "groq_daily_limit",
                "at_question": number,
                "completed": len(per_question),
                "resume_offset": number - 1,
                "retry_in": limit.retry_in,
                "detail": limit.detail,
            }
            print(f"\nSTOPPED: {_stop_summary(stopped)}\n  {limit.detail}")
            return per_question, stopped
        per_question.append(record)
        # Between questions only: outside every timed run, so it never counts as latency.
        if args.sleep and index < len(questions):
            await asyncio.sleep(args.sleep)
    return per_question, None


def _results(meta: dict, cfg: dict, per_question: list[dict]) -> dict:
    """The saved run: summaries are always recomputed from the per-question records."""
    return {
        "meta": {**meta, "completed_questions": len(per_question)},
        "config": cfg,
        "retrieval": _summarize_retrieval(per_question, tuple(cfg["retrieval"]["modes"])),
        "answers": _summarize_answers(per_question, tuple(cfg["answers"]["pipelines"])),
        "per_question": per_question,
    }


def _save(results: dict, json_path: Path) -> None:
    md_path = json_path.with_suffix(".md")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    markdown = _markdown(results)
    md_path.write_text(markdown, encoding="utf-8")
    print("\n" + markdown)
    print(f"Saved {json_path} and {md_path}")


def _load_results(path: Path) -> dict:
    try:
        results = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        sys.exit(f"{path}: {exc}")
    if not isinstance(results, dict) or not {"meta", "config", "per_question"} <= results.keys():
        sys.exit(f"{path}: not a results file written by run_eval")
    meta = results["meta"]
    stopped = meta.get("stopped")
    if stopped and stopped.get("stage") == "judging":
        # Written before judging stops had their own key.
        meta["stopped"], meta["judging_stopped"] = None, stopped
    meta.setdefault("judging_stopped", None)
    return results


def _approximate_contexts(runs: list[dict], user_id: str) -> None:
    """Stand-in context texts for ``runs`` saved without them: each cited page's full text, once."""
    docs = load_user_docs(user_id)
    for run in runs:
        pages = list(dict.fromkeys((name, page) for name, page in run["contexts"]))
        run["context_texts"] = [docs[name].page_text(page) for name, page in pages if name in docs]
        run["context_texts_approximate"] = True


def _score_only(args: argparse.Namespace) -> None:
    path: Path = args.score_only
    results = _load_results(path)
    meta, cfg, per_question = results["meta"], results["config"], results["per_question"]
    pipelines = tuple(cfg["answers"]["pipelines"])
    if not pipelines:
        sys.exit(f"{path} has no answers to score (a --retrieval-only run)")
    only_unscored = not args.rescore

    previous_judge = cfg["answers"].get("judge_model")
    scored = any(
        run["faithfulness"] is not None or run["answer_relevancy"] is not None
        for q in per_question for run in q["answers"].values()
    )
    if only_unscored and scored and previous_judge and previous_judge != args.judge_model:
        sys.exit(
            f"{path} already has scores from {previous_judge}; judging the rest with {args.judge_model} "
            f"would mix judges. Pass --judge-model {previous_judge}, or --rescore to judge every answer again."
        )

    targets = [q["answers"][p] for p in pipelines for q in _score_targets(per_question, p, only_unscored)]
    if not targets:
        print(f"{path}: every answer already has scores; pass --rescore to judge them again.")
        return
    missing = [run for run in targets if "context_texts" not in run]
    if missing and not args.approx_contexts:
        sys.exit(
            f"{path}: {len(missing)} answer(s) were saved without their context texts (results written "
            "before they were kept), so RAGAS can't see what the model saw. Re-run those questions, or "
            "pass --approx-contexts to judge against the whole cited pages from the index (approximate)."
        )
    if missing:
        _approximate_contexts(missing, args.user_id or cfg["user_id"])

    started = datetime.now(timezone.utc)
    judging_stopped = _judge(per_question, pipelines, args, only_unscored)
    cfg["answers"].update(
        judge_model=args.judge_model,
        ragas_metrics=["faithfulness", "answer_relevancy"],
        answer_relevancy_strictness=ANSWER_RELEVANCY_STRICTNESS,
    )
    meta["judging_stopped"] = judging_stopped
    meta.setdefault("scoring_runs", []).append({
        "timestamp": started.isoformat(timespec="seconds"),
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "git": _git_info(),
        "judge_model": args.judge_model,
        "rescore": args.rescore,
        "approximate_contexts": len(missing),
    })
    _save(_results(meta, cfg, per_question), path)
    if judging_stopped:
        sys.exit(3)


# Config that must be identical for runs to be merged into one result.
_MERGE_MUST_MATCH = (
    ("questions_sha256",), ("scope",), ("user_id",), ("chroma_database",), ("retrieval_only",),
    ("retrieval",), ("answers", "pipelines"), ("answers", "retrieval_mode"),
)


def _config_value(cfg: dict, keys: tuple[str, ...]) -> Any:
    for key in keys:
        cfg = cfg.get(key) if isinstance(cfg, dict) else None
    return cfg


def _merge(args: argparse.Namespace) -> None:
    runs = [(path, _load_results(path)) for path in args.merge]
    for path, results in runs:
        if not results["per_question"]:
            sys.exit(f"{path} has no completed questions to merge")
    runs.sort(key=lambda run: min(q["number"] for q in run[1]["per_question"]))
    first, configs = runs[0][1], [results["config"] for _, results in runs]

    problems = [
        f"{'.'.join(keys)} differs: "
        + ", ".join(f"{path.name}={_config_value(r['config'], keys)!r}" for path, r in runs)
        for keys in _MERGE_MUST_MATCH
        if any(_config_value(c, keys) != _config_value(configs[0], keys) for c in configs)
    ]
    seen: dict[int, Path] = {}
    for path, results in runs:
        for q in results["per_question"]:
            if q["number"] in seen:
                problems.append(f"question {q['number']} is in both {seen[q['number']].name} and {path.name}")
            seen[q["number"]] = path
    # Scores are only comparable when one judge gave them all.
    judges = {
        r["config"]["answers"].get("judge_model") for _, r in runs
        if any(
            run["faithfulness"] is not None or run["answer_relevancy"] is not None
            for q in r["per_question"] for run in q["answers"].values()
        )
    }
    if len(judges) > 1:
        problems.append(
            f"scores come from different judges ({', '.join(sorted(map(str, judges)))}); "
            "re-judge one run with --score-only --rescore --judge-model <the other's> first"
        )
    if problems:
        sys.exit("can't merge:\n  " + "\n  ".join(problems))

    per_question = sorted((q for _, r in runs for q in r["per_question"]), key=lambda q: q["number"])
    numbers = [q["number"] for q in per_question]
    gaps = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    if gaps:
        print(f"! questions {gaps} are in none of the runs; the merged result skips them")
    answer_models = {json.dumps(c["answers"].get("answer_models")) for c in configs}
    if len(answer_models) > 1:
        print(f"! runs used different answer models: {', '.join(sorted(answer_models))}")
    if len({json.dumps(c.get("versions"), sort_keys=True) for c in configs}) > 1:
        print("! runs used different package versions; each run's are kept under meta.merged_from")

    cfg = json.loads(json.dumps(first["config"]))
    judged = next(
        (c for c in configs if c["answers"].get("judge_model") in judges),
        next((c for c in configs if c["answers"].get("judge_model")), cfg),
    )
    cfg.update(
        questions=len(per_question),
        questions_by_type=dict(Counter(q["type"] for q in per_question)),
        offset=numbers[0] - 1,
        limit=None,
    )
    cfg["answers"].update(
        answer_models=sorted({m for c in configs for m in c["answers"].get("answer_models", [])}),
        **{k: judged["answers"].get(k) for k in ("judge_model", "ragas_metrics", "answer_relevancy_strictness")},
    )

    metas = [results["meta"] for _, results in runs]
    commits = {m["git"]["commit"] for m in metas}
    if len(commits) > 1:
        print("! runs came from different commits; each run's is kept under meta.merged_from")
    meta = {
        "timestamp": metas[0]["timestamp"],
        "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "duration_s": round(sum(m.get("duration_s") or 0 for m in metas), 1),
        "git": {
            "commit": next(iter(commits)) if len(commits) == 1 else None,
            "dirty": any(m["git"]["dirty"] for m in metas),
        },
        # The run with the last questions decides whether the merged run is complete.
        "stopped": metas[-1].get("stopped"),
        "judging_stopped": next((m["judging_stopped"] for m in reversed(metas) if m["judging_stopped"]), None),
        "merged_from": [
            {
                "file": path.name,
                "first": min(q["number"] for q in r["per_question"]),
                "last": max(q["number"] for q in r["per_question"]),
                "timestamp": r["meta"]["timestamp"],
                "git": r["meta"]["git"],
                "offset": r["config"].get("offset"),
                "pacing": r["config"].get("pacing"),
                "versions": r["config"].get("versions"),
                "stopped": r["meta"].get("stopped"),
                "judging_stopped": r["meta"]["judging_stopped"],
            }
            for path, r in runs
        ],
    }
    _save(_results(meta, cfg, per_question), args.out_dir / "results.merged.json")


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
    parser.add_argument(
        "--offset", type=int, default=0, metavar="N",
        help="skip the first N questions, e.g. to resume a run that stopped at question N+1",
    )
    parser.add_argument("--limit", type=int, help="run at most N questions (after --offset)")
    parser.add_argument(
        "--retrieval-only", action="store_true",
        help="only the retrieval comparison: no answer pipelines, no RAGAS, no Groq calls",
    )
    parser.add_argument("--skip-ragas", action="store_true", help="no judge calls; faithfulness/relevancy stay empty")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Groq model RAGAS uses as the judge")
    parser.add_argument("--judge-workers", type=int, default=2, help="concurrent RAGAS judge calls (Groq rate limits)")
    parser.add_argument("--judge-timeout", type=int, default=180)
    parser.add_argument("--delay", type=float, default=0.0, help="seconds to wait after each pipeline run")
    parser.add_argument(
        "--sleep", type=float, default=0.0, metavar="SECONDS",
        help="seconds to wait between questions, to stay under Groq rate limits",
    )
    saved = parser.add_mutually_exclusive_group()
    saved.add_argument(
        "--score-only", type=Path, metavar="RESULTS_JSON",
        help="judge a saved run's answers with RAGAS (no retrieval or answering) and rewrite its .json/.md",
    )
    saved.add_argument(
        "--merge", type=Path, nargs="+", metavar="RESULTS_JSON",
        help="combine partial runs into results.merged.json/.md in --out-dir, recomputing every summary",
    )
    parser.add_argument("--rescore", action="store_true", help="with --score-only: judge already-scored answers again")
    parser.add_argument(
        "--approx-contexts", action="store_true",
        help="with --score-only: judge answers saved without context texts against their cited pages' full text",
    )
    args = parser.parse_args()
    if args.score_only:
        if args.skip_ragas:
            parser.error("--score-only only runs RAGAS; drop --skip-ragas")
        _score_only(args)
        return
    if args.merge:
        if len(args.merge) < 2:
            parser.error("--merge needs at least two results files")
        _merge(args)
        return
    if args.rescore or args.approx_contexts:
        parser.error("--rescore and --approx-contexts only apply to --score-only")
    if not args.user_id:
        parser.error("provide --user-id or set RAG_EVAL_USER_ID")
    if args.offset < 0:
        parser.error("--offset must be 0 or more")
    if args.retrieval_only:
        args.pipelines, args.skip_ragas = (), True
        if not args.modes:
            parser.error("--retrieval-only with no --modes leaves nothing to run")

    questions_bytes = args.questions.read_bytes()
    try:
        questions = validate_questions(json.loads(questions_bytes))
    except (QuestionError, json.JSONDecodeError) as exc:
        sys.exit(f"{args.questions}: {exc}")
    if not questions:
        sys.exit(f"{args.questions} has no questions; draft some with `python -m eval.generate_questions`")
    if args.offset >= len(questions):
        sys.exit(f"--offset {args.offset} skips every question; {args.questions} has {len(questions)}")
    questions = questions[args.offset:]
    if args.limit:
        questions = questions[: args.limit]

    _install_instrumentation()
    answer_retrieval = _set_answer_retrieval(args.answer_retrieval)
    started = datetime.now(timezone.utc)
    per_question, stopped = asyncio.run(_collect(args, questions))
    # RAGAS runs its own event loop, so it goes after (not inside) asyncio.run.
    judging_stopped = None
    if args.pipelines and not args.skip_ragas and per_question:
        if stopped and args.judge_model in _answer_models:
            # Groq's quotas are per model: this one is the quota that just ran out.
            judging_stopped = {
                "stage": "judging",
                "reason": "groq_daily_limit",
                "retry_in": stopped["retry_in"],
                "detail": f"not attempted: the judge {args.judge_model} is an answer model and hit its daily limit",
            }
            print(f"RAGAS skipped: the judge {args.judge_model} is the answer model whose daily limit stopped the run.")
        else:
            if stopped:
                print(
                    f"Judging the {len(per_question)} completed question(s) anyway: "
                    f"the judge {args.judge_model} has its own daily quota."
                )
            judging_stopped = _judge(per_question, args.pipelines, args)

    meta = {
        "timestamp": started.isoformat(timespec="seconds"),
        "duration_s": round((datetime.now(timezone.utc) - started).total_seconds(), 1),
        "git": _git_info(),
        "stopped": stopped,
        "judging_stopped": judging_stopped,
    }
    # A resumed run gets its own files rather than overwriting the run it continues.
    stem = f"results.offset{args.offset}" if args.offset else "results"
    _save(
        _results(meta, _config(args, questions, questions_bytes, answer_retrieval), per_question),
        args.out_dir / f"{stem}.json",
    )
    if stopped or judging_stopped:
        sys.exit(3)  # partial results: lets scripts tell this apart from a complete run (or a usage error, 2)


if __name__ == "__main__":
    main()
