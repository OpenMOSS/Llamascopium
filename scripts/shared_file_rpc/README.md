c# Shared-file RPC bridge

This bridge is a workaround for environments where the FastAPI backend works
on the remote machine, but the notebook or proxy URL returns `403 Forbidden`.

```text
local frontend -> local bridge http://127.0.0.1:24577
               -> SSH
               -> remote shared-directory request JSON
               -> backend-side worker
               -> remote backend http://127.0.0.1:3000
               -> shared-directory response JSON
               -> local bridge -> frontend
```

The local bridge and remote worker must use the same absolute `--queue-dir`.

## 1. Start the backend on the remote machine

From the remote Llamascopium checkout:

```bash
uv run uvicorn server.app:app \
  --host 0.0.0.0 \
  --port 3000 \
  --env-file server/.env
```

The worker accesses this backend at `127.0.0.1:3000` on the remote machine.

## 2. Start the worker on the remote machine

Run this in a second remote terminal from the remote Llamascopium checkout:

```bash
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/dev/Language-Model-SAEs
uv run python scripts/shared_file_rpc/backend_file_worker.py \
  --queue-dir /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/dev/Language-Model-SAEs/.shared_file_rpc \
  --backend-url http://127.0.0.1:3000
```

## 3. Start the SSH bridge on the local machine

From the local checkout, use the same remote queue path:

```bash
uv run python scripts/shared_file_rpc/local_ssh_bridge.py \
  --ssh-host root@qz-ssh-wstunnel \
  --queue-dir /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/dev/Language-Model-SAEs/.shared_file_rpc \
  --host 127.0.0.1 \
  --port 24577
```

Verify the bridge locally:

```bash
curl http://127.0.0.1:24577/docs
```

All command-line settings also have environment-variable equivalents for the
most commonly reused values:

```bash
export SHARED_FILE_RPC_SSH_HOST=root@qz-ssh-wstunnel
export SHARED_FILE_RPC_DIR=/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/dev/Language-Model-SAEs/.shared_file_rpc
export SHARED_FILE_RPC_BACKEND_URL=http://127.0.0.1:3000
```

## 4. Point the frontend to the bridge

Set the backend URL in `ui/.env`:

```bash
BACKEND_URL=http://127.0.0.1:24577
```

Then start the frontend:

```bash
cd ui
bun dev
```

## Notes

- This is intended for interactive debugging, not high-throughput service.
- Each request and response is copied through SSH, so it is slower than a
  direct HTTP tunnel.
- The bridge deletes completed response files by default. Pass
  `--no-delete-responses` to retain them for debugging.
- Pass `--archive` to the worker to retain successful request files.
- For a timeout, check the remote backend with
  `curl http://127.0.0.1:3000/docs`, inspect the worker logs, and inspect
  `/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/dev/Language-Model-SAEs/.shared_file_rpc/failed`.
