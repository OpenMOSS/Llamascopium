src_base="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4_bigger/sae"
dst_base="/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/result_BT4/sae/k_30_e_16"
mkdir -p "$dst_base"

for i in $(seq 0 14); do
    src="${src_base}/lc0_L${i}M_16x_k30_lr8e-04_auxk_sparseadam"
    
    if [ -d "$src" ]; then
        echo "Moving: $src → $dst_base"
        mv "$src" "$dst_base/"
    else
        echo "Warning: directory does not exist → $src"
    fi
done
for i in $(seq 0 14); do
    old="${dst_base}/lc0_L${i}M_16x_k30_lr8e-04_auxk_sparseadam"
    new="${dst_base}/L${i}"

    if [ -d "$old" ]; then
        echo "Renaming: $old → $new"
        mv "$old" "$new"
    else
        echo "Warning: directory not found → $old"
    fi
done



CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29300 \
  exp/analyze_lc0_mlp_BT4.py \
    --layer 0 \
    --n_tokens 10000



# qz run


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate 
export PYTHONPATH=/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/src:$PYTHONPATH  
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in $(seq 0 1); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done


cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate  
export PYTHONPATH=/inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N/src:$PYTHONPATH 
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in $(seq 2 3); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
      
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in $(seq 4 5); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in $(seq 6 7); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in $(seq 8 9); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in $(seq 10 11); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in $(seq 12 13); do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs-N
. .venv/bin/activate   
LOGDIR="$(pwd)/logs/analyze_mlp"
mkdir -p "$LOGDIR"

for L in 14; do
  echo "===> layer $L"
  CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$((29440+L)) \
    exp/analyze_lc0_mlp_BT4.py \
      --layer "$L" \
    > "$LOGDIR/BT4_mlp_l${L}_analyze.log" 2>&1
done



CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=29020 \
    exp/analyze_lc0_mlp_BT4.py \
      --layer 0 \
      --k 30 \
      --exp_factor 16