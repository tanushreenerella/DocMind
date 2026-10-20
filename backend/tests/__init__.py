"""Test bootstrap: keep the suite fully offline.

``services.embedder`` opens a Chroma Cloud client and creates its collection at
import time, and ``core.config`` refuses to import without the Chroma env vars.
Importing it here, before any test module, with a fake CloudClient and dummy
credentials means tests never need real credentials and can never touch a real
Chroma Cloud database (dev or prod), whatever is in the shell or ``.env``.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

# Unconditional on purpose: a real key from .env/the shell must not leak in.
os.environ["CHROMA_API_KEY"] = "test-api-key"
os.environ["CHROMA_TENANT"] = "test-tenant"
os.environ["CHROMA_DATABASE"] = "test-database"
os.environ.setdefault("JWT_SECRET_KEY", "test-secret")
# Fixed so tests know what to send as x-proxy-secret (see PROXY_HEADERS).
os.environ["PROXY_SECRET"] = "test-proxy-secret"
PROXY_HEADERS = {"x-proxy-secret": "test-proxy-secret"}
# Retrieval flags off unless a test patches them on, so a developer's .env
# can't make tests download and run the real reranker model.
os.environ["HYBRID_SEARCH_ENABLED"] = "false"
os.environ["RERANK_ENABLED"] = "false"
# In-process counters; never a real Redis from .env.
os.environ["RATE_LIMIT_STORAGE_URI"] = "memory://"
# Never a real R2 bucket from .env; anything that isn't patched to a
# FakeStorage falls back to a throwaway local directory.
for _name in ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
    os.environ[_name] = ""
os.environ["LOCAL_OBJECTS_PATH"] = tempfile.mkdtemp(prefix="docmind-test-objects-")

_fake_collection = MagicMock()
_fake_collection.name = "documents"
_fake_collection.configuration_json = {"hnsw": {"space": "cosine"}}
_fake_client = MagicMock()
_fake_client.get_or_create_collection.return_value = _fake_collection

with patch("chromadb.CloudClient", return_value=_fake_client):
    from services import embedder  # noqa: F401,E402
