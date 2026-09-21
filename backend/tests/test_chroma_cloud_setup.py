"""Chroma Cloud wiring: fail-fast config and the cosine-distance guard."""

import importlib
import os
import unittest
from unittest.mock import MagicMock, patch

from core import config
from services import embedder


def _collection_with(configuration_json: dict) -> MagicMock:
    collection = MagicMock()
    collection.name = "documents"
    collection.configuration_json = configuration_json
    return collection


class CosineGuardTests(unittest.TestCase):
    def test_accepts_cosine_hnsw(self) -> None:
        embedder._assert_cosine_space(
            _collection_with({"hnsw": {"space": "cosine"}})
        )

    def test_accepts_cosine_spann(self) -> None:
        embedder._assert_cosine_space(
            _collection_with({"spann": {"space": "cosine"}})
        )

    def test_rejects_l2(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "'l2'"):
            embedder._assert_cosine_space(
                _collection_with({"hnsw": {"space": "l2"}})
            )

    def test_rejects_unreported_space(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "None"):
            embedder._assert_cosine_space(_collection_with({}))

    def test_collection_is_requested_with_cosine_configuration(self) -> None:
        # Module import already ran against the fake client from tests/__init__.
        call = embedder.chroma_client.get_or_create_collection.call_args
        self.assertEqual(call.kwargs["name"], "documents")
        self.assertEqual(
            call.kwargs["configuration"], {"hnsw": {"space": "cosine"}}
        )


class ConfigFailFastTests(unittest.TestCase):
    def tearDown(self) -> None:
        # Leave core.config in its normal (fully configured) state.
        importlib.reload(config)

    def test_missing_vars_are_all_named(self) -> None:
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("CHROMA_")
        }
        env["CHROMA_API_KEY"] = "k"  # only tenant + database missing
        # load_dotenv must not refill the vars from a developer's real .env.
        with (
            patch.dict(os.environ, env, clear=True),
            patch("dotenv.load_dotenv"),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                importlib.reload(config)

        message = str(ctx.exception)
        self.assertIn("CHROMA_TENANT", message)
        self.assertIn("CHROMA_DATABASE", message)
        self.assertNotIn("CHROMA_API_KEY", message)


if __name__ == "__main__":
    unittest.main()
