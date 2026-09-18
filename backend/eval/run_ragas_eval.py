"""Evaluate the RAG pipeline against editable question/answer fixtures.

Run from the backend directory, for example:
    RAG_EVAL_USER_ID=<user-id> python eval/run_ragas_eval.py
"""

import argparse
import asyncio
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

from datasets import Dataset
from langchain_core.embeddings import Embeddings
from langchain_groq import ChatGroq
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from core.config import GROQ_API_KEY
from services.embedder import search
from services.embeddings import get_embedding, get_embeddings
from services.rag import answer_query

DEFAULT_QUESTIONS_FILE = Path(__file__).with_name("test_questions.json")
DEFAULT_RESULTS_FILE = Path(__file__).with_name("results.json")
METRICS = ("faithfulness", "answer_relevancy", "context_precision")


class GeminiEmbeddings(Embeddings):
    """Expose the application's Gemini embeddings through LangChain's interface."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return asyncio.run(get_embeddings(texts))

    def embed_query(self, text: str) -> list[float]:
        return asyncio.run(get_embedding(text))

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await get_embeddings(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return await get_embedding(text)


def _build_ragas_evaluators() -> tuple[
    LangchainLLMWrapper,
    LangchainEmbeddingsWrapper,
]:
    # Match the production answer model so judge behavior tracks user-facing output.
    judge_llm = LangchainLLMWrapper(
        ChatGroq(
           model="qwen/3.8-27b",
            api_key=GROQ_API_KEY,
            temperature=0,
            reasoning_format="hidden",
        )
    )
    judge_embeddings = LangchainEmbeddingsWrapper(GeminiEmbeddings())
    return judge_llm, judge_embeddings


def _load_questions(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        questions = json.load(source)

    if not isinstance(questions, list):
        raise ValueError("Test questions JSON must contain a list of examples")
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Example {index} must be an object")
        if not item.get("question") or not item.get("ground_truth"):
            raise ValueError(f"Example {index} needs question and ground_truth fields")
    return questions


async def _collect_examples(
    questions: list[dict[str, Any]],
    user_id: str,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for item in questions:
        question = item["question"]
        doc_id = item.get("doc_id")
        contexts = await search(
            question,
            n_results=6,
            doc_id=doc_id,
            user_id=user_id,
        )
        response = await answer_query(
            question,
            conversation_history=[],
            doc_id=doc_id,
            user_id=user_id,
        )
        examples.append(
            {
                "question": question,
                "ground_truth": item["ground_truth"],
                "answer": response["answer"],
                "contexts": [hit["chunk_text"] for hit in contexts],
                "doc_id": doc_id,
                "has_answer": response["has_answer"],
            }
        )
    return examples


def _score_examples(examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    dataset = Dataset.from_list(
        [
            {
                key: example[key]
                for key in ("question", "ground_truth", "answer", "contexts")
            }
            for example in examples
        ]
    )
    judge_llm, judge_embeddings = _build_ragas_evaluators()
    evaluation = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
        llm=judge_llm,
        embeddings=judge_embeddings,
        raise_exceptions=False,
    )
    rows = evaluation.to_pandas().to_dict(orient="records")
    for example, row in zip(examples, rows):
        example["metrics"] = {
            metric: _finite_float(row.get(metric))
            for metric in METRICS
        }
    return examples


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _summarize(examples: list[dict[str, Any]]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for metric in METRICS:
        scores = [
            item["metrics"][metric]
            for item in examples
            if item["metrics"][metric] is not None
        ]
        summary[metric] = sum(scores) / len(scores) if scores else None
    return summary


def _print_summary(summary: dict[str, float | None], total: int) -> None:
    print(f"RAGAS results ({total} examples)")
    print(f"{'Metric':<22} {'Score':>8}")
    print("-" * 31)
    for metric in METRICS:
        score = summary[metric]
        display = f"{score:.4f}" if score is not None else "n/a"
        print(f"{metric:<22} {display:>8}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the DocMind RAG pipeline with RAGAS")
    parser.add_argument("--user-id", default=os.environ.get("RAG_EVAL_USER_ID"))
    parser.add_argument("--questions-file", type=Path, default=DEFAULT_QUESTIONS_FILE)
    parser.add_argument("--results-file", type=Path, default=DEFAULT_RESULTS_FILE)
    args = parser.parse_args()

    if not args.user_id:
        parser.error("provide --user-id or set RAG_EVAL_USER_ID")

    examples = asyncio.run(_collect_examples(_load_questions(args.questions_file), args.user_id))
    scored_examples = _score_examples(examples)
    summary = _summarize(scored_examples)
    results = {"summary": summary, "examples": scored_examples}

    args.results_file.parent.mkdir(parents=True, exist_ok=True)
    with args.results_file.open("w", encoding="utf-8") as destination:
        json.dump(results, destination, indent=2, ensure_ascii=False)

    _print_summary(summary, len(scored_examples))
    print(f"Results saved to {args.results_file}")


if __name__ == "__main__":
    main()
