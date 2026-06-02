#!/usr/bin/env bash
set -euo pipefail
# gen 1d training activation
CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29620 --nproc_per_node=2 exp/gen_evo2_tc.py \
    --layers 26 \
    --total-tokens 10_100_000

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29630 --nproc_per_node=2 exp/gen_evo2_tc_2d.py \
    --layer 26 \
    --total-tokens 1_100_000


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29620 --nproc_per_node=1 exp/gen_evo2_tc.py \
  --layers 26 \
  --output-dir /inspire/hdd/project/reasoning/public/activations/evo2_7b/opengenome2/activations \
  --context-size 65536 \
  --crop-mode random \
  --model-batch-size 1 \
  --batch-size 8 \
  --total-tokens 501000000

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29620 --nproc_per_node=8 exp/gen_evo2_tc.py \
  --layers 26 \
  --output-dir /inspire/hdd/project/reasoning/public/activations/evo2_7b/opengenome2/activations \
  --context-size 8192 \
  --crop-mode random \
  --model-batch-size 1 \
  --batch-size 1 \
  --buffer-size 1 \
  --total-tokens 1000000
    
# gen 2d analyzing activation
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29630 --nproc_per_node=8 exp/gen_evo2_tc_2d.py \
    --layer 26 \
    --context-size 32768 \
    --output-dir /inspire/hdd/project/reasoning/public/activations/evo2_7b/opengenome2/activations \
    --model-batch-size 1\
    --batch-size 1 \
    --total-tokens 100_100_000