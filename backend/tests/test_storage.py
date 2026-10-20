"""services.storage: key layout, local and R2 backends, and backend selection."""

import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from services import storage

USER = "aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"
DOC = "11111111-2222-3333-4444-555555555555"


class KeyTests(unittest.TestCase):
    def test_keys_are_namespaced_by_user_and_document(self) -> None:
        self.assertEqual(storage.document_prefix(USER, DOC), f"users/{USER}/{DOC}/")
        self.assertEqual(storage.pdf_key(USER, DOC), f"users/{USER}/{DOC}/original.pdf")
        self.assertEqual(
            storage.page_image_key(USER, DOC, f"{DOC}_p2.jpg"),
            f"users/{USER}/{DOC}/pages/{DOC}_p2.jpg",
        )

    def test_unsafe_segments_are_rejected(self) -> None:
        for user_id, doc_id in (("..", DOC), (USER, "../other"), (USER, "a/b"), ("", DOC)):
            with self.subTest(user_id=user_id, doc_id=doc_id):
                with self.assertRaises(ValueError):
                    storage.document_prefix(user_id, doc_id)
        with self.assertRaises(ValueError):
            storage.page_image_key(USER, DOC, "../../x.jpg")


class LocalStorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = tempfile.mkdtemp()
        self.store = storage.LocalStorage(self.root)

    def tearDown(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)

    def test_put_get_delete_roundtrip(self) -> None:
        key = storage.pdf_key(USER, DOC)
        self.store.put(key, b"%PDF-1.4 data", "application/pdf")

        obj = self.store.get(key)
        self.assertEqual(b"".join(obj.body), b"%PDF-1.4 data")
        self.assertEqual(obj.size, len(b"%PDF-1.4 data"))
        self.assertEqual(obj.content_type, "application/pdf")

        self.store.delete(key)
        self.assertIsNone(self.store.get(key))
        self.store.delete(key)  # idempotent

    def test_delete_prefix_removes_only_that_document(self) -> None:
        other_doc = "99999999-2222-3333-4444-555555555555"
        self.store.put(storage.pdf_key(USER, DOC), b"a", "application/pdf")
        self.store.put(storage.page_image_key(USER, DOC, f"{DOC}_p1.jpg"), b"b", "image/jpeg")
        self.store.put(storage.pdf_key(USER, other_doc), b"c", "application/pdf")

        self.store.delete_prefix(storage.document_prefix(USER, DOC))

        self.assertIsNone(self.store.get(storage.pdf_key(USER, DOC)))
        self.assertIsNone(self.store.get(storage.page_image_key(USER, DOC, f"{DOC}_p1.jpg")))
        self.assertIsNotNone(self.store.get(storage.pdf_key(USER, other_doc)))
        self.store.delete_prefix(storage.document_prefix(USER, DOC))  # idempotent

    def test_keys_cannot_escape_the_root(self) -> None:
        for key in ("../outside.txt", "users/../../outside.txt"):
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    self.store.put(key, b"x", "text/plain")
        self.assertFalse(os.path.exists(os.path.join(os.path.dirname(self.root), "outside.txt")))


class R2StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MagicMock()
        self.client.exceptions.NoSuchKey = type("NoSuchKey", (Exception,), {})
        self.store = storage.R2Storage(self.client, "bucket")

    def test_put_writes_to_the_private_bucket(self) -> None:
        self.store.put("users/u/d/original.pdf", b"data", "application/pdf")

        self.client.put_object.assert_called_once_with(
            Bucket="bucket", Key="users/u/d/original.pdf", Body=b"data", ContentType="application/pdf"
        )
        # No ACL argument: objects inherit the private bucket policy.
        self.assertNotIn("ACL", self.client.put_object.call_args.kwargs)

    def test_get_streams_the_body(self) -> None:
        body = MagicMock()
        body.iter_chunks.return_value = iter([b"ab", b"cd"])
        self.client.get_object.return_value = {
            "Body": body, "ContentLength": 4, "ContentType": "image/jpeg",
        }

        obj = self.store.get("users/u/d/pages/x.jpg")

        self.assertEqual(b"".join(obj.body), b"abcd")
        self.assertEqual((obj.size, obj.content_type), (4, "image/jpeg"))
        self.client.get_object.assert_called_once_with(Bucket="bucket", Key="users/u/d/pages/x.jpg")

    def test_get_missing_key_is_none(self) -> None:
        self.client.get_object.side_effect = self.client.exceptions.NoSuchKey()
        self.assertIsNone(self.store.get("users/u/d/original.pdf"))

    def test_delete_prefix_deletes_every_listed_page(self) -> None:
        self.client.get_paginator.return_value.paginate.return_value = [
            {"Contents": [{"Key": "users/u/d/original.pdf"}, {"Key": "users/u/d/pages/1.jpg"}]},
            {"Contents": [{"Key": "users/u/d/pages/2.jpg"}]},
            {},  # an empty page carries no Contents at all
        ]

        self.store.delete_prefix("users/u/d/")

        self.client.get_paginator.return_value.paginate.assert_called_once_with(
            Bucket="bucket", Prefix="users/u/d/"
        )
        deleted = [
            [o["Key"] for o in call.kwargs["Delete"]["Objects"]]
            for call in self.client.delete_objects.call_args_list
        ]
        self.assertEqual(
            deleted,
            [["users/u/d/original.pdf", "users/u/d/pages/1.jpg"], ["users/u/d/pages/2.jpg"]],
        )


class BackendSelectionTests(unittest.TestCase):
    R2_VARS = ("R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")

    def _config(self, **values: str):
        settings = {name: values.get(name, "") for name in self.R2_VARS}
        return patch.multiple(storage.config, **settings)

    def test_no_r2_vars_uses_local_disk(self) -> None:
        with self._config(), patch("boto3.client") as boto_client:
            self.assertIsInstance(storage.create_storage(), storage.LocalStorage)
        boto_client.assert_not_called()

    def test_all_r2_vars_use_r2(self) -> None:
        values = {
            "R2_ENDPOINT_URL": "https://acct.r2.cloudflarestorage.com",
            "R2_ACCESS_KEY_ID": "id",
            "R2_SECRET_ACCESS_KEY": "secret",
            "R2_BUCKET": "docmind",
        }
        with self._config(**values), patch("boto3.client") as boto_client:
            backend = storage.create_storage()

        self.assertIsInstance(backend, storage.R2Storage)
        kwargs = boto_client.call_args.kwargs
        self.assertEqual(boto_client.call_args.args, ("s3",))
        self.assertEqual(kwargs["endpoint_url"], values["R2_ENDPOINT_URL"])
        self.assertEqual(kwargs["aws_access_key_id"], "id")
        self.assertEqual(kwargs["aws_secret_access_key"], "secret")

    def test_partial_r2_config_fails_loudly(self) -> None:
        with self._config(R2_ENDPOINT_URL="https://x", R2_BUCKET="docmind"):
            with self.assertRaises(RuntimeError) as ctx:
                storage.create_storage()
        self.assertIn("R2_ACCESS_KEY_ID", str(ctx.exception))
        self.assertIn("R2_SECRET_ACCESS_KEY", str(ctx.exception))
        self.assertNotIn("R2_BUCKET", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
