"""Shared test doubles."""

from services.storage import StoredObject


class FakeStorage:
    """In-memory stand-in for services.storage backends (R2 or local disk)."""

    def __init__(self) -> None:
        self.objects: dict[str, tuple[bytes, str]] = {}

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
