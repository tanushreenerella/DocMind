"""Backfill Chroma chunk metadata with the document owner's user_id.

Run from the repository root after deploying the metadata-filtering change:
    backend/venv/Scripts/python scripts/backfill_chroma_user_metadata.py --apply

Without --apply the script only reports what it would change.
"""

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1] / "backend"
sys.path.insert(0, str(BACKEND_DIR))

from core.database import connect, disconnect, get_pool  # noqa: E402
from services.embedder import collection  # noqa: E402


async def backfill(apply: bool) -> None:
    await connect()
    try:
        rows = await get_pool().fetch(
            "SELECT doc_id, user_id FROM documents"
        )
        owners = {
            str(row["doc_id"]): str(row["user_id"])
            for row in rows
        }

        stored = collection.get(include=["metadatas"])
        updates: list[tuple[str, dict]] = []
        skipped = 0

        for chunk_id, metadata in zip(
            stored.get("ids", []),
            stored.get("metadatas", []),
        ):
            if not metadata:
                skipped += 1
                continue

            owner = owners.get(str(metadata.get("doc_id", "")))
            if not owner:
                skipped += 1
                continue

            if metadata.get("user_id") == owner:
                continue

            patched_metadata = dict(metadata)
            patched_metadata["user_id"] = owner
            updates.append((chunk_id, patched_metadata))

        print(
            f"Found {len(updates)} chunks to backfill; "
            f"skipped {skipped} chunks without a mapped document owner."
        )
        if not apply or not updates:
            print("Dry run complete." if not apply else "Nothing to update.")
            return

        batch_size = 100
        for start in range(0, len(updates), batch_size):
            batch = updates[start : start + batch_size]
            collection.update(
                ids=[chunk_id for chunk_id, _ in batch],
                metadatas=[metadata for _, metadata in batch],
            )

        print(f"Backfilled user_id metadata for {len(updates)} chunks.")
    finally:
        await disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backfill Chroma chunk user_id metadata from Postgres."
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write updates; without this flag the script is a dry run.",
    )
    args = parser.parse_args()
    asyncio.run(backfill(args.apply))
