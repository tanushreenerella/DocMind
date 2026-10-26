# DocMind eval — 2026-10-22T15:58:20+00:00

Commit `52f31e1169 (uncommitted changes)` · 41 questions (26 factual, 8 multi_part, 7 unanswerable) · scope: all

## Retrieval (34 answerable questions, page-level relevance)

| Mode | Recall@5 | MRR@10 | nDCG@10 | p50 ms | p95 ms | Errors |
|---|---:|---:|---:|---:|---:|---:|
| vector | 0.912 | 0.659 | 0.732 | 770 | 825 | 0 |
| bm25 | 0.912 | 0.719 | 0.773 | 5 | 9 | 0 |
| hybrid | 0.897 | 0.705 | 0.763 | 776 | 830 | 0 |
| hybrid_rerank | 0.926 | 0.720 | 0.773 | 841 | 897 | 0 |

Latency = embedding + queries (+ fusion, rerank) per question. No LLM calls.

## Config

```json
{
  "questions_file": "C:\\Users\\tanus\\OneDrive - Manipal University Jaipur\\BFAI Assignment\\bfai-assessment\\backend\\eval\\questions.json",
  "questions_sha256": "ae5c3551127d8b9db54cff61e2542ef78b9d9bdc1a1c440cdb13e28602d6aecc",
  "questions": 41,
  "questions_by_type": {
    "factual": 26,
    "multi_part": 8,
    "unanswerable": 7
  },
  "offset": 0,
  "limit": null,
  "retrieval_only": true,
  "scope": "all",
  "pacing": {
    "sleep_between_questions_s": 0.0,
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
    "pipelines": [],
    "retrieval_mode": "hybrid_rerank",
    "answer_models": [],
    "judge_model": null,
    "ragas_metrics": [],
    "answer_relevancy_strictness": null
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
