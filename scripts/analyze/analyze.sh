# analyzing

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/Evo2/Llamascopium
. .venv/bin/activate
export PYTHONPATH=/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/src:$PYTHONPATH  
LOGDIR="$(pwd)/logs/analyze/evo2_tc/8x64k"
mkdir -p "$LOGDIR"

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29440 \
  exp/analyze_evo2_tc.py \
    --layer 26 \
    --k 64 \
    --exp_factor 8 \
    --n_tokens 1_000_000 \
  > "$LOGDIR/evo2_tc_l26_analyze.log" 2>&1