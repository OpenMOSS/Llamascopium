#!/usr/bin/env python3
"""Forward shared-file RPC requests to a backend HTTP service.

Run this on the same machine as the backend service. It watches a shared
directory for JSON request files, forwards each request to the backend HTTP
service, and writes JSON response files back to the shared directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import sys
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

STOP = False


def _handle_signal(_signum: int, _frame: Any) -> None:
    global STOP
    STOP = True


def ensure_queue_dirs(queue_dir: Path) -> dict[str, Path]:
    dirs = {
        "requests": queue_dir / "requests",
        "processing": queue_dir / "processing",
        "responses": queue_dir / "responses",
        "failed": queue_dir / "failed",
        "archive": queue_dir / "archive",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def load_request(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"request payload must be an object: {path}")
    if not payload.get("id"):
        raise ValueError(f"request missing id: {path}")
    return payload


def forward_to_backend(payload: dict[str, Any], backend_url: str, timeout: float) -> dict[str, Any]:
    method = str(payload.get("method", "GET")).upper()
    path = str(payload.get("path", "/"))
    query_string = str(payload.get("query_string", ""))
    body_base64 = payload.get("body_base64") or ""
    body = base64.b64decode(body_base64) if body_base64 else None

    if not path.startswith("/"):
        path = "/" + path
    url = backend_url.rstrip("/") + path
    if query_string:
        url += "?" + query_string.lstrip("?")

    raw_headers = payload.get("headers") or {}
    headers = {
        str(key): str(value)
        for key, value in raw_headers.items()
        if str(key).lower() not in {"host", "content-length", "connection", "accept-encoding"}
    }
    if body is not None and "content-type" not in {key.lower() for key in headers}:
        headers["content-type"] = "application/octet-stream"

    request = urllib.request.Request(url=url, data=body, headers=headers, method=method)

    # The backend normally runs on localhost. Do not let cluster-wide HTTP proxy
    # variables redirect this request away from the worker machine.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            response_body = response.read()
            response_headers = dict(response.headers.items())
            status_code = response.status
    except urllib.error.HTTPError as exc:
        response_body = exc.read()
        response_headers = dict(exc.headers.items()) if exc.headers else {}
        status_code = exc.code

    return {
        "id": payload["id"],
        "ok": 200 <= int(status_code) < 400,
        "status_code": int(status_code),
        "headers": response_headers,
        "body_base64": base64.b64encode(response_body).decode("ascii"),
        "completed_at": time.time(),
    }


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False)
    os.replace(tmp_path, path)


def process_one(
    request_path: Path,
    dirs: dict[str, Path],
    backend_url: str,
    timeout: float,
    archive: bool,
) -> None:
    processing_path = dirs["processing"] / request_path.name
    try:
        os.replace(request_path, processing_path)
    except FileNotFoundError:
        return

    try:
        payload = load_request(processing_path)
        response_payload = forward_to_backend(payload, backend_url=backend_url, timeout=timeout)
        response_path = dirs["responses"] / f"{payload['id']}.json"
        atomic_write_json(response_path, response_payload)
        if archive:
            shutil.move(str(processing_path), str(dirs["archive"] / processing_path.name))
        else:
            processing_path.unlink(missing_ok=True)
        print(
            f"[ok] {payload.get('method')} {payload.get('path')} -> "
            f"{response_payload['status_code']} ({payload['id']})",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - keep the worker alive for later requests
        failure = {
            "id": processing_path.stem,
            "ok": False,
            "status_code": 599,
            "headers": {"content-type": "application/json"},
            "body_base64": base64.b64encode(
                json.dumps(
                    {
                        "error": "backend_file_worker_failed",
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
            ).decode("ascii"),
            "completed_at": time.time(),
        }
        atomic_write_json(dirs["responses"] / f"{processing_path.stem}.json", failure)
        shutil.move(str(processing_path), str(dirs["failed"] / processing_path.name))
        print(f"[failed] {processing_path.name}: {exc}", file=sys.stderr, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-dir",
        default=os.environ.get("SHARED_FILE_RPC_DIR", ".shared_file_rpc"),
        help="Shared queue directory visible to both machines.",
    )
    parser.add_argument(
        "--backend-url",
        default=os.environ.get("SHARED_FILE_RPC_BACKEND_URL", "http://127.0.0.1:3000"),
        help="Backend URL reachable from this worker machine.",
    )
    parser.add_argument("--poll-interval", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--archive", action="store_true", help="Keep processed request files under archive/.")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    queue_dir = Path(args.queue_dir).expanduser().resolve()
    dirs = ensure_queue_dirs(queue_dir)
    print("[start] shared-file RPC worker", flush=True)
    print(f"[queue] {queue_dir}", flush=True)
    print(f"[backend] {args.backend_url}", flush=True)

    while not STOP:
        request_files = sorted(dirs["requests"].glob("*.json"), key=lambda path: path.stat().st_mtime)
        if not request_files:
            time.sleep(args.poll_interval)
            continue
        for request_path in request_files:
            if STOP:
                break
            process_one(
                request_path=request_path,
                dirs=dirs,
                backend_url=args.backend_url,
                timeout=args.timeout,
                archive=args.archive,
            )

    print("[stop] shared-file RPC worker stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
