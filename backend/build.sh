#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
pip install -r requirements.txt
# Fetch the reranker now, into the project dir (core.config.FLASHRANK_CACHE_DIR),
# which Render keeps from the build; at runtime FlashRank then finds it and
# skips the download. Defaults must match core/config.py.
python -c '
import os
from flashrank import Ranker
Ranker(
    model_name=os.environ.get("RERANK_MODEL", "ms-marco-TinyBERT-L-2-v2"),
    cache_dir=os.environ.get("FLASHRANK_CACHE_DIR", os.path.join(os.getcwd(), ".cache", "flashrank")),
)
print("[build] reranker model cached")
'
