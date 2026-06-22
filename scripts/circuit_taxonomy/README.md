# Circuit Taxonomy Runner

## Start

From repo root:

```bash
uv run uvicorn server.app:app --host 0.0.0.0 --port 3000 --env-file server/.env
```

If the backend is on a remote machine, use that machine's address from local commands:

```text
http://REMOTE_HOST:3000
```

If remote port 3000 is not directly reachable, create a tunnel on a free local port:

```bash
ssh -L 3001:127.0.0.1:3000 USER@REMOTE_HOST
```

Then use:

```text
http://127.0.0.1:3001
```

If `ssh` prints `open failed: connect failed: Connection refused`, the backend is not reachable as
`127.0.0.1:3000` from the SSH host. Check on the SSH host:

```bash
curl http://127.0.0.1:3000/docs
```

If the SSH host is only a gateway, tunnel to the backend host as seen from that gateway:

```bash
ssh -L 3001:BACKEND_HOST:3000 USER@GATEWAY_HOST
```

ssh -N -L 3001:127.0.0.1:3000 root@qz-ssh-wstunnel

In another terminal:

```bash
cd ui
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/circuit-taxonomy
```

Use `Export Evidence` on the page to download the same JSONL evidence used by the runner.

## Generate 100 Codex Proposals

From repo root:

```bash
python scripts/circuit_taxonomy/taxonomy_runner.py run \
  --base-url http://127.0.0.1:3001/ \
  --limit 100 \
  --evidence-output outputs/circuit_taxonomy/evidence.jsonl \
  --proposals-output outputs/circuit_taxonomy/proposals.jsonl \
  --chunk-size 5
```

Then copy `outputs/circuit_taxonomy/proposals.jsonl` into:

```text
Circuit Taxonomy Annotation -> Import Proposals
```

## Evidence Only

```bash
python scripts/circuit_taxonomy/taxonomy_runner.py extract \
  --base-url http://REMOTE_HOST:3000 \
  --limit 100 \
  --output outputs/circuit_taxonomy/evidence.jsonl
```
