"""Shared test doubles."""

from services.storage import StoredObject


class FakeStorage:
    """In-memory stand-in for services.storage backends (R2 or local disk).

    Behaves like local disk (uploads go through the API) unless built with
    ``direct_uploads=True``, which makes it hand out presigned URLs like R2.
    """

    def __init__(self, direct_uploads: bool = False) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.direct_uploads = direct_uploads

    def size(self, key: str) -> int | None:
        return len(self.objects[key][0]) if key in self.objects else None

    def download(self, key: str, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self.objects[key][0])

    def presign_put(self, key: str, content_type: str, expires_seconds: int) -> str:
        if not self.direct_uploads:
            raise NotImplementedError("local storage has no presigned URLs")
        return f"https://r2.example/{key}?ct={content_type}&expires={expires_seconds}&sig=x"

    def put(self, key: str, data: bytes, content_type: str) -> None:
        self.objects[key] = (data, content_type)

    def get(self, key: str) -> StoredObject | None:
        if key not in self.objects:
            return None
        data, content_type = self.objects[key]
        return StoredObject(body=iter([data]), size=len(data), content_type=content_type)

    def delete(self, key: str) -> None:
        self.objects.pop(key, None)

    def delete_prefix(self, prefix: str) -> None:
        for key in [k for k in self.objects if k.startswith(prefix)]:
            del self.objects[key]

    def keys(self, prefix: str = "") -> list[str]:
        return sorted(k for k in self.objects if k.startswith(prefix))
