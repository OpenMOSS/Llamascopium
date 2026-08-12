# Shared-file RPC bridge

This is a workaround for environments where the FastAPI backend works on the
remote machine (`curl http://127.0.0.1:3000/docs`) but the notebook/proxy URL
returns `403 Forbidden`.

The data flow is:

```text
local browser -> local bridge http://127.0.0.1:3000
              -> ssh root@qz-ssh-wstunnel
              -> shared directory request JSON
              -> backend-side worker
              -> remote backend http://127.0.0.1:3000
              -> shared directory response JSON
              -> local bridge -> browser
```

## 1. Start the backend on the backend machine

```bash
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
uv run uvicorn server.app:app --host 127.0.0.1 --port 3000 --env-file server/.env
```

It is fine to use `--host 0.0.0.0`, but this bridge only requires
`127.0.0.1:3000` on the backend machine.

## 2. Start the file worker on the backend machine

Run this in a second terminal on the backend machine:

```bash
cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
python scripts/shared_file_rpc/backend_file_worker.py \
  --queue-dir /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/.shared_file_rpc \
  --backend-url http://127.0.0.1:3000
```

## 3. Start the local SSH bridge on your local machine

Run this from your local checkout:

```bash
python scripts/shared_file_rpc/local_ssh_bridge.py \
  --ssh-host root@qz-ssh-wstunnel \
  --queue-dir /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/.shared_file_rpc \
  --host 127.0.0.1 \
  --port 3000
```

Then verify locally:

```bash
curl http://127.0.0.1:3000/docs
```

## 4. Point the frontend to the local bridge

In `ui/.env`, use:

```bash
VITE_BACKEND_URL=http://127.0.0.1:3000
```

Then run local frontend:

```bash
cd ui
bun dev
```

Open the local Vite URL in your browser.

## Notes

- This bridge is for interactive debugging, not high-throughput service.
- Large endpoints work, but every request/response is copied through SSH, so
  it will be slower than a direct HTTP tunnel.
- If a request times out, check:
  - backend service: `curl http://127.0.0.1:3000/docs` on backend machine;
  - worker terminal logs;
  - queue files under `.shared_file_rpc/failed`.
