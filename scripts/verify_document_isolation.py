"""Manual end-to-end check for cross-user document isolation.

Starts with a real User A JWT and User B JWT, uploads a document as A, waits
for indexing, then confirms B receives only the generic 403 response when
asking /api/chat about A's doc_id.
"""

import argparse
import json
import mimetypes
import time
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def request_json(
    url: str,
    method: str,
    token: str,
    body: bytes | None = None,
    content_type: str | None = None,
) -> tuple[int, dict]:
    headers = {"Authorization": f"Bearer {token}"}
    if content_type:
        headers["Content-Type"] = content_type
    request = Request(url, data=body, headers=headers, method=method)
    try:
        with urlopen(request) as response:
            return response.status, json.loads(response.read().decode())
    except HTTPError as error:
        payload = error.read().decode()
        return error.code, json.loads(payload) if payload else {}


def multipart_file(file_path: Path) -> tuple[bytes, str]:
    boundary = f"----document-isolation-{uuid.uuid4().hex}"
    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
    payload = [
        f"--{boundary}\r\n".encode(),
        (
            f'Content-Disposition: form-data; name="files"; '
            f'filename="{file_path.name}"\r\n'
        ).encode(),
        f"Content-Type: {mime_type}\r\n\r\n".encode(),
        file_path.read_bytes(),
        f"\r\n--{boundary}--\r\n".encode(),
    ]
    return b"".join(payload), f"multipart/form-data; boundary={boundary}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--user-a-token", required=True)
    parser.add_argument("--user-b-token", required=True)
    parser.add_argument("--document", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=120)
    args = parser.parse_args()

    body, content_type = multipart_file(args.document)
    status, upload = request_json(
        f"{args.base_url.rstrip('/')}/api/upload",
        "POST",
        args.user_a_token,
        body,
        content_type,
    )
    if status != 200 or not upload.get("jobs"):
        raise SystemExit(f"Upload failed: HTTP {status} {upload}")

    doc_id = upload["jobs"][0]["doc_id"]
    deadline = time.monotonic() + args.timeout_seconds
    while time.monotonic() < deadline:
        status, job = request_json(
            f"{args.base_url.rstrip('/')}/api/status/{doc_id}",
            "GET",
            args.user_a_token,
        )
        if status == 200 and job.get("status") == "complete":
            break
        if status != 200 or job.get("status") == "error":
            raise SystemExit(f"Indexing failed: HTTP {status} {job}")
        time.sleep(2)
    else:
        raise SystemExit("Timed out waiting for indexing")

    status, response = request_json(
        f"{args.base_url.rstrip('/')}/api/chat",
        "POST",
        args.user_b_token,
        json.dumps({
            "question": "Summarize this document.",
            "doc_id": doc_id,
            "conversation_history": [],
        }).encode(),
        "application/json",
    )
    if status != 403 or response != {"detail": "Document access denied"}:
        raise SystemExit(f"Isolation check failed: HTTP {status} {response}")

    print("PASS: User B received generic 403 and no chunks/citations were returned.")


if __name__ == "__main__":
    main()
