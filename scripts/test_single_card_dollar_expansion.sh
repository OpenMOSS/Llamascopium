#!/bin/sh

set -eu

. ~/.local/bin/env

REPO_DIR="${REPO_DIR:-/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/llamascope2/dev/Language-Model-SAEs}"
MASTER_PORT="${MASTER_PORT:-29617}"

cd "$REPO_DIR"

nnode=1
nproc=1
log_dir="${LOG_DIR:-exp/train/logs/shell-variable-test}"
log_file="$log_dir/single-card-dollar-expansion.log"

mkdir -p "$log_dir"
[ -f "$log_file" ] && mv "$log_file" "${log_file%.log}_$(date +%Y%m%d_%H%M%S).bak"

printf 'nnode=%s\n' "$nnode"
printf 'nproc=%s\n' "$nproc"
printf 'log_file=%s\n' "$log_file"

uv run torchrun \
  --nnodes="$nnode" \
  --nproc_per_node="$nproc" \
  --master_addr=127.0.0.1 \
  --master_port="$MASTER_PORT" \
  --no-python /bin/bash -c \
  'printf "PASS local_rank=%s world_size=%s cuda_visible_devices=%s\n" "$LOCAL_RANK" "$WORLD_SIZE" "${CUDA_VISIBLE_DEVICES:-<unset>}"' \
  > "$log_file" 2>&1

grep -q '^PASS local_rank=0 world_size=1 ' "$log_file"
cat "$log_file"
printf 'PASS: nnode, nproc, MASTER_PORT, and log_file expanded correctly.\n'
