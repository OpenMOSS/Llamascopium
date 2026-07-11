#!/usr/bin/env python3
"""Expose a local HTTP bridge backed by shared-file RPC over SSH.

Run this on your local machine. The browser/frontend talks to this local HTTP
server, and this bridge writes request files plus reads response files on the
remote shared filesystem through SSH.

Example:
    python scripts/shared_file_rpc/local_ssh_bridge.py \\
      --ssh-host root@qz-ssh-wstunnel \\
      --queue-dir /inspire/hdd/global_user/.../chess-SAEs-N/.shared_file_rpc \\
      --host 127.0.0.1 --port 3000
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import shlex
import subprocess
import time
import uuid
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware


def shell_quote(value: str) -> str:
    return shlex.quote(value)


def run_ssh(ssh_host: str, remote_command: str, *, stdin: bytes | None = None, timeout: float = 30.0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["ssh", ssh_host, remote_command],
        input=stdin,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout,
        check=False,
    )


def write_remote_request(ssh_host: str, queue_dir: str, request_id: str, payload: dict[str, Any], timeout: float) -> None:
    request_dir = f"{queue_dir.rstrip('/')}/requests"
    tmp_path = f"{request_dir}/{request_id}.json.tmp"
    final_path = f"{request_dir}/{request_id}.json"
    remote_command = (
        f"mkdir -p {shell_quote(request_dir)} "
        f"{shell_quote(queue_dir.rstrip('/') + '/responses')} "
        f"{shell_quote(queue_dir.rstrip('/') + '/processing')} "
        f"{shell_quote(queue_dir.rstrip('/') + '/failed')} && "
        f"cat > {shell_quote(tmp_path)} && mv {shell_quote(tmp_path)} {shell_quote(final_path)}"
    )
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    result = run_ssh(ssh_host, remote_command, stdin=data, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="replace"))


def read_remote_response(ssh_host: str, queue_dir: str, request_id: str, timeout: float) -> dict[str, Any] | None:
    response_path = f"{queue_dir.rstrip('/')}/responses/{request_id}.json"
    remote_command = f"test -f {shell_quote(response_path)} && cat {shell_quote(response_path)}"
    result = run_ssh(ssh_host, remote_command, timeout=timeout)
    if result.returncode != 0 or not result.stdout:
        return None
    return json.loads(result.stdout.decode("utf-8"))


def delete_remote_response(ssh_host: str, queue_dir: str, request_id: str, timeout: float) -> None:
    response_path = f"{queue_dir.rstrip('/')}/responses/{request_id}.json"
    remote_command = f"rm -f {shell_quote(response_path)}"
    run_ssh(ssh_host, remote_command, timeout=timeout)


def create_app(args: argparse.Namespace) -> FastAPI:
    app = FastAPI(title="Shared File RPC SSH Bridge")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"])
    async def proxy(path: str, request: Request) -> Response:
        if request.method == "OPTIONS":
            return Response(status_code=204)

        request_id = uuid.uuid4().hex
        body = await request.body()
        headers = {
            k: v
            for k, v in request.headers.items()
            if k.lower() not in {"host", "content-length", "connection", "accept-encoding"}
        }
        payload = {
            "id": request_id,
            "method": request.method,
            "path": "/" + path,
            "query_string": request.url.query,
            "headers": headers,
            "body_base64": base64.b64encode(body).decode("ascii") if body else "",
            "created_at": time.time(),
        }

        write_remote_request(
            ssh_host=args.ssh_host,
            queue_dir=args.queue_dir,
            request_id=request_id,
            payload=payload,
            timeout=args.ssh_timeout,
        )

        deadline = time.time() + args.request_timeout
        response_payload: dict[str, Any] | None = None
        while time.time() < deadline:
            response_payload = read_remote_response(
                ssh_host=args.ssh_host,
                queue_dir=args.queue_dir,
                request_id=request_id,
                timeout=args.ssh_timeout,
            )
            if response_payload is not None:
                break
            time.sleep(args.poll_interval)

        if response_payload is None:
            return Response(
                json.dumps(
                    {
                        "error": "shared_file_rpc_timeout",
                        "request_id": request_id,
                        "timeout_seconds": args.request_timeout,
                    },
                    ensure_ascii=False,
                ),
                status_code=504,
                media_type="application/json",
            )

        if args.delete_responses:
            delete_remote_response(args.ssh_host, args.queue_dir, request_id, timeout=args.ssh_timeout)

        response_body = base64.b64decode(response_payload.get("body_base64") or "")
        response_headers = response_payload.get("headers") or {}
        passthrough_headers = {
            str(k): str(v)
            for k, v in response_headers.items()
            if str(k).lower()
            not in {
                "content-length",
                "content-encoding",
                "transfer-encoding",
                "connection",
                "server",
                "date",
            }
        }
        return Response(
            content=response_body,
            status_code=int(response_payload.get("status_code", 502)),
            headers=passthrough_headers,
        )

    return app


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssh-host", default=os.environ.get("SHARED_FILE_RPC_SSH_HOST", "root@qz-ssh-wstunnel"))
    parser.add_argument(
        "--queue-dir",
        default=os.environ.get("SHARED_FILE_RPC_DIR", ".shared_file_rpc"),
        help="Remote shared queue directory path.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=3000)
    parser.add_argument("--poll-interval", type=float, default=0.25)
    parser.add_argument("--request-timeout", type=float, default=300.0)
    parser.add_argument("--ssh-timeout", type=float, default=30.0)
    parser.add_argument("--delete-responses", action="store_true", default=True)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(create_app(args), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
