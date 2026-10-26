# DocMind eval — 2026-10-24T21:54:52+00:00

Commit `52f31e1169 (uncommitted changes)` · 20 questions (13 factual, 4 multi_part, 3 unanswerable) · scope: all

> **Partial run — Groq daily limit reached at question 20; Groq says try again in 8m4.704s. 19 question(s) completed and saved; resume with --offset 19.**

## Retrieval (17 answerable questions, page-level relevance)

| Mode | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Errors |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.824 | 0.614 | 0.695 | 1,705 | 2,757 | 0 |
| bm25 | 0.824 | 0.633 | 0.693 | 3 | 25 | 0 |
| hybrid | 0.794 | 0.606 | 0.670 | 1,708 | 2,761 | 0 |
| hybrid_rerank | 0.882 | 0.622 | 0.689 | 1,720 | 2,774 | 0 |

Latency = embedding + queries (+ fusion, rerank) per question. No LLM calls.

## Answers (retrieval: hybrid_rerank, judge: openai/gpt-oss-120b)

| Pipeline | Faithfulness | Answer relevancy | Unanswerable declined | Answerable declined | Errors |
|---|---:|---:|---:|---:|---:|
| basic | — (n=0) | — (n=0) | 2/2 | 3/17 | 0 q (0) |
| agentic | — (n=0) | — (n=0) | 2/2 | 1/17 | 1 q (1) |

### Latency and cost

| Pipeline | p50 / p95 end-to-end ms | … excl. 429 wait | p50 / p95 retrieval ms | 429 retries (questions) | 429 wait total s | Tokens/q in / out | LLM calls/q |
|---|---:|---:|---:|---:|---:|---:|---:|
| basic | 1,418 / 5,843 | 1,418 / 5,843 | 1,043 / 1,400 | 0 (0) | 0.0 | 1279 / 45 | 1.0 |
| agentic | 15,193 / 35,941 | 2,375 / 5,457 | 1,056 / 4,707 | 31 (14) | 252.4 | 3638 / 170 | 3.5 |

Declines are detected from has_answer=false or refusal phrasing (see decline_reason per question in results.json). Retrieval time is summed over a run's search calls; agentic searches run concurrently, so it can exceed wall-clock time. 429 wait is the wall-clock time the Groq client spent sleeping before retrying rate-limited calls (overlapping waits counted once); "excl. 429 wait" subtracts it from each run.

## Config

```json
{
  "questions_file": "C:\\Users\\tanus\\OneDrive - Manipal University Jaipur\\BFAI Assignment\\bfai-assessment\\backend\\eval\\questions.json",
  "questions_sha256": "ae5c3551127d8b9db54cff61e2542ef78b9d9bdc1a1c440cdb13e28602d6aecc",
  "questions": 20,
  "questions_by_type": {
    "factual": 13,
    "multi_part": 4,
    "unanswerable": 3
  },
  "offset": 0,
  "limit": 20,
  "retrieval_only": false,
  "scope": "all",
  "pacing": {
    "sleep_between_questions_s": 20.0,
    "delay_after_pipeline_s": 0.0,
    "groq_max_retries": 2
  },
  "user_id": "e36d9dd4-cf9f-4e43-a321-88702b23297d",
  "chroma_database": "docmind-dev",
  "retrieval": {
    "modes": [
      "vector",
      "bm25",
      "hybrid",
      "hybrid_rerank"
    ],
    "recall_k": 5,
    "rank_k": 10,
    "relevance": "page-level: doc_name matches and page in expected_pages",
    "embedding_model": "models/gemini-embedding-001",
    "chunk_size": 500,
    "chunk_overlap": 50,
    "rrf_k": 60,
    "rerank_model": "ms-marco-TinyBERT-L-2-v2",
    "rerank_candidates": 10
  },
  "answers": {
    "pipelines": [
      "basic",
      "agentic"
    ],
    "retrieval_mode": "hybrid_rerank",
    "answer_models": [
      "qwen/qwen3.8-27b"
    ],
    "judge_model": "openai/gpt-oss-120b",
    "ragas_metrics": [
      "faithfulness",
      "answer_relevancy"
    ],
    "answer_relevancy_strictness": 1
  },
  "versions": {
    "python": "3.11.0",
    "ragas": "0.4.3",
    "langchain-groq": "0.3.8",
    "groq": "0.37.1",
    "chromadb": "1.5.9",
    "flashrank": "0.2.10"
  }
}
```
