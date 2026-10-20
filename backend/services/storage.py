"""Object storage for original PDFs and page images.

Cloudflare R2 (S3 API, private bucket) when all four R2_* vars are set, else a
local directory with the same key layout. Every key lives under
``users/{user_id}/{doc_id}/`` so one prefix delete removes a whole document.

Objects are never exposed by URL: routes check ownership, then stream them
through ``get``. All calls block; run them via ``asyncio.to_thread``.
"""

import os
import re
import shutil
from dataclasses import dataclass
from typing import Iterator, Protocol

from core import config

_STREAM_CHUNK_BYTES = 64 * 1024
# user ids and doc ids are UUIDs; page images are "{doc_id}_p{n}.jpg".
_SAFE_SEGMENT = re.compile(r"^[A-Za-z0-9_.-]+$")


@dataclass
class StoredObject:
    body: Iterator[bytes]
    size: int | None
    content_type: str


class Storage(Protocol):
    def put(self, key: str, data: bytes, content_type: str) -> None: ...
    def get(self, key: str) -> StoredObject | None: ...
    def delete(self, key: str) -> None: ...
    def delete_prefix(self, prefix: str) -> None: ...


# ── keys ────────────────────────────────────────────────────────────────────

def _segment(value: str) -> str:
    if not _SAFE_SEGMENT.fullmatch(value) or value in {".", ".."}:
        raise ValueError(f"unsafe storage key segment: {value!r}")
    return value


def document_prefix(user_id: str, doc_id: str) -> str:
    return f"users/{_segment(user_id)}/{_segment(doc_id)}/"


def pdf_key(user_id: str, doc_id: str) -> str:
    return document_prefix(user_id, doc_id) + "original.pdf"


def page_image_key(user_id: str, doc_id: str, image_filename: str) -> str:
    return document_prefix(user_id, doc_id) + "pages/" + _segment(image_filename)


# ── backends ────────────────────────────────────────────────────────────────

class R2Storage:
    def __init__(self, client, bucket: str) -> None:
        self._client = client
        self._bucket = bucket

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self._client.put_object(Bucket=self._bucket, Key=key, Body=data, ContentType=content_type)

    def get(self, key: str) -> StoredObject | None:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
        except self._client.exceptions.NoSuchKey:
            return None
        return StoredObject(
            body=obj["Body"].iter_chunks(_STREAM_CHUNK_BYTES),
            size=obj.get("ContentLength"),
            content_type=obj.get("ContentType") or "application/octet-stream",
        )

    def delete(self, key: str) -> None:
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def delete_prefix(self, prefix: str) -> None:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if keys:  # list pages hold at most 1000 keys, the delete_objects limit
                self._client.delete_objects(
                    Bucket=self._bucket, Delete={"Objects": keys, "Quiet": True}
                )


class LocalStorage:
    _CONTENT_TYPES = {".pdf": "application/pdf", ".jpg": "image/jpeg"}

    def __init__(self, root: str) -> None:
        self._root = os.path.realpath(root)

    def _path(self, key: str) -> str:
        path = os.path.realpath(os.path.join(self._root, *key.strip("/").split("/")))
        if os.path.commonpath([path, self._root]) != self._root:
            raise ValueError(f"storage key escapes root: {key!r}")
        return path

    def put(self, key: str, data: bytes, content_type: str) -> None:
        path = self._path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)

    def get(self, key: str) -> StoredObject | None:
        path = self._path(key)
        if not os.path.isfile(path):
            return None

        def chunks() -> Iterator[bytes]:
            with open(path, "rb") as f:
                while chunk := f.read(_STREAM_CHUNK_BYTES):
                    yield chunk

        content_type = self._CONTENT_TYPES.get(os.path.splitext(path)[1].lower())
        return StoredObject(
            body=chunks(),
            size=os.path.getsize(path),
            content_type=content_type or "application/octet-stream",
        )

    def delete(self, key: str) -> None:
        try:
            os.remove(self._path(key))
        except FileNotFoundError:
            pass

    def delete_prefix(self, prefix: str) -> None:
        shutil.rmtree(self._path(prefix), ignore_errors=True)


def create_storage() -> Storage:
    r2_settings = {
        "R2_ENDPOINT_URL": config.R2_ENDPOINT_URL,
        "R2_ACCESS_KEY_ID": config.R2_ACCESS_KEY_ID,
        "R2_SECRET_ACCESS_KEY": config.R2_SECRET_ACCESS_KEY,
        "R2_BUCKET": config.R2_BUCKET,
    }
    missing = [name for name, value in r2_settings.items() if not value]
    if len(missing) == len(r2_settings):
        print(f"[storage] R2 not configured; using local disk at {config.LOCAL_OBJECTS_PATH}")
        return LocalStorage(config.LOCAL_OBJECTS_PATH)
    if missing:
        # Half-configured R2 is a deploy mistake; silently writing to an
        # ephemeral local disk instead would lose every upload.
        raise RuntimeError("Missing R2 environment variable(s): " + ", ".join(missing))

    import boto3
    from botocore.config import Config

    client = boto3.client(
        "s3",
        endpoint_url=config.R2_ENDPOINT_URL,
        aws_access_key_id=config.R2_ACCESS_KEY_ID,
        aws_secret_access_key=config.R2_SECRET_ACCESS_KEY,
        region_name="auto",
        config=Config(signature_version="s3v4", retries={"max_attempts": 3, "mode": "standard"}),
    )
    print(f"[storage] using R2 bucket {config.R2_BUCKET}")
    return R2Storage(client, config.R2_BUCKET)


_storage: Storage | None = None


def get_storage() -> Storage:
    global _storage
    if _storage is None:
        _storage = create_storage()
    return _storage


def put(key: str, data: bytes, content_type: str) -> None:
    get_storage().put(key, data, content_type)


def get(key: str) -> StoredObject | None:
    return get_storage().get(key)


def delete(key: str) -> None:
    get_storage().delete(key)


def delete_prefix(prefix: str) -> None:
    get_storage().delete_prefix(prefix)
