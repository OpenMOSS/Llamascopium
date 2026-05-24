# Demo
CUDA_VISIBLE_DEVICES=0,1 torchrun --nnodes=1 --nproc_per_node=2 --master-port=29440 \
  exp/train_tc_evo2.py \
    --lr 1e-4 \
    --layer 26 \
    --tp 2 \
    --dp 1 \
    --k 64 \
    --exp-factor 8 \
    --total-training-tokens 10_000_000


# Sweeping

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/Evo2/Llamascopium
. .venv/bin/activate
export CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1

LOGDIR="$(pwd)/logs/evo2_tc/8x64k"
mkdir -p "$LOGDIR"

for LR in 1e-4 1e-5 1e-6; do
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nnodes=1 --nproc_per_node=4 --master-port=29440 \
    exp/train_tc_evo2.py \
      --lr $LR \
      --layer 26 \
      --tp 4 \
      --dp 1 \
      --k 64 \
      --exp-factor 8 \
      --total-training-tokens 10_000_000 \
      > $LOGDIR/lr_$LR.log 2>&1 &
done

