#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
set -a
source server/.env
set +a

mapfile -t selection < <(.venv/bin/python - <<'PY'
import os

from llamascopium.database import MongoClient, MongoDBConfig

client = MongoClient(MongoDBConfig(mongo_uri=os.environ["MONGO_URI"], mongo_db=os.environ["MONGO_DB"]))
target = "qwen3-1.7b-lorsa-32x-topk128-layer20"
record = client.get_sae(target, "qwen3-1.7b")
if record is None or record.cfg.hook_point_out != "blocks.20.hook_attn_out" or record.cfg.d_sae <= 55953:
    raise SystemExit(f"Target {target} is missing or does not contain L20A-55953")
sae_set_name = "qwen3-1.7b-full-32x-k128-final"
sae_set = client.get_sae_set(sae_set_name)
if sae_set is None or sae_set.sae_series != "qwen3-1.7b" or target not in sae_set.sae_names:
    raise SystemExit(f"{sae_set_name} is missing or does not contain {target}")
print(sae_set_name)
print(target)
PY
)
if [[ ${#selection[@]} -ne 2 ]]; then
  echo "Could not resolve the 32x SAE set and target" >&2
  exit 1
fi
sae_set="${selection[0]}"
target_sae="${selection[1]}"

echo "Scanning $target_sae #55953 with $sae_set"
.venv/bin/llamascopium inhibitory-global-weights \
  --series qwen3-1.7b \
  --sae-set "$sae_set" \
  --target-sae "$target_sae" \
  --feature-id 55953 \
  --dataset qwen3 \
  --max-samples 10000 \
  --depth 3 \
  --expansion-width 3 \
  --device cuda \
  --output results/L20A-55953-inhibitory.json
