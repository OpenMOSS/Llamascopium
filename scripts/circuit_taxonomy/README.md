# Circuit Taxonomy Runner

## Start

From repo root:

```bash
python -m uvicorn server.app:app --port 24577 --env-file server/.env
```

If `uvicorn` is missing:

```bash
python -m pip install "uvicorn[standard]"
python -m uvicorn server.app:app --port 24577 --env-file server/.env
```

In another terminal:

```bash
cd ui
npm run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173/circuit-taxonomy
```

## Generate 100 Codex Proposals

From repo root:

```bash
python scripts/circuit_taxonomy/taxonomy_runner.py run \
  --base-url http://127.0.0.1:24577 \
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
  --base-url http://127.0.0.1:24577 \
  --limit 100 \
  --output outputs/circuit_taxonomy/evidence.jsonl
```
