# DocMind eval — 2026-10-22T13:09:34+00:00

Commit `7a9cad2768 (uncommitted changes)` · 5 questions (5 factual) · scope: all

## Retrieval (5 answerable questions, page-level relevance)

| Mode | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Errors |
|---|---:|---:|---:|---:|---:|---:|
| vector | 1.000 | 0.633 | 0.710 | 831 | 1,584 | 0 |
| bm25 | 1.000 | 0.700 | 0.742 | 6 | 7 | 0 |
| hybrid | 1.000 | 0.717 | 0.762 | 836 | 1,590 | 0 |
| hybrid_rerank | 1.000 | 0.567 | 0.649 | 907 | 1,679 | 0 |

Latency = embedding + queries (+ fusion, rerank) per question. No LLM calls.

## Answers (retrieval: hybrid_rerank, judge: RAGAS skipped)

| Pipeline | Faithfulness | Answer relevancy | Unanswerable declined | Answerable declined | p50 / p95 end-to-end ms | p50 / p95 retrieval ms | Tokens/q in / out | LLM calls/q | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| basic | — (n=0) | — (n=0) | 0/0 | 1/5 | 7,458 / 9,272 | 1,081 / 1,107 | 1086 / 33 | 1.0 | 0 q (0) |
| agentic | — (n=0) | — (n=0) | 0/0 | 0/5 | 22,345 / 29,796 | 1,142 / 2,112 | 2948 / 146 | 3.2 | 0 q (0) |

Declines are detected from has_answer=false or refusal phrasing (see decline_reason per question in results.json). Retrieval time is summed over a run's search calls; agentic searches run concurrently, so it can exceed wall-clock time.

## Config

```json
{
  "questions_file": "C:\\Users\\tanus\\OneDrive - Manipal University Jaipur\\BFAI Assignment\\bfai-assessment\\backend\\eval\\questions.json",
  "questions_sha256": "ae5c3551127d8b9db54cff61e2542ef78b9d9bdc1a1c440cdb13e28602d6aecc",
  "questions": 5,
  "questions_by_type": {
    "factual": 5
  },
  "limit": 5,
  "scope": "all",
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
    "judge_model": null,
    "ragas_metrics": []
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
