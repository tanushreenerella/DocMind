"""Test bootstrap: keep the suite fully offline.

``services.embedder`` opens a Chroma Cloud client and creates its collection at
import time, and ``core.config`` refuses to import without the Chroma env vars.
Importing it here, before any test module, with a fake CloudClient and dummy
credentials means tests never need real credentials and can never touch a real
Chroma Cloud database (dev or prod), whatever is in the shell or ``.env``.
"""

import os
from unittest.mock import MagicMock, patch

# Unconditional on purpose: a real key from .env/the shell must not leak in.
os.environ["CHROMA_API_KEY"] = "test-api-key"
os.environ["CHROMA_TENANT"] = "test-tenant"
os.environ["CHROMA_DATABASE"] = "test-database"
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")

_fake_collection = MagicMock()
_fake_collection.name = "documents"
_fake_collection.configuration_json = {"hnsw": {"space": "cosine"}}
_fake_client = MagicMock()
_fake_client.get_or_create_collection.return_value = _fake_collection

with patch("chromadb.CloudClient", return_value=_fake_client):
    from services import embedder  # noqa: F401,E402
