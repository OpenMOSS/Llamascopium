uv add chess
uv add onnx 
uv add onnx2torch
uv add onnxruntime
uv add chex
uv add grain
uv add dm-haiku
uv add optax
uv add orbax
uv add apache_beam
uv add -U "jax[cuda12]"   

uv pip install -e .

cd /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs
. .venv/bin/activate

#搓数据

python /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/exp/00dataset_test/01dataset_merge/deduplicate_merge_shuffle_batch.py
python /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/exp/00dataset_test/01dataset_merge/deduplicate_merge_shuffle_batch_best_only.py

# merge and shuffle data
python exp/00dataset_test/01dataset_merge/merge_and_shuffle_json.py \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/json_data/puzzle_demo.json \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/chess_analysis_0001.json \
    -o /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/json_data/shuffled/merged_test.json -s 42

python exp/00dataset_test/01dataset_merge/merge_and_shuffle_json.py \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/standard01.json \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/standard02.json \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/standard03.json \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/standard04.json \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/standard05.json \
    /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/standard06.json \
    -o /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/lichess_standard_rated/standard.json -s 42

python 

# gen
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --master_port=29600 --nproc_per_node=4 exp/gen_linear_policy.py

CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=29610 --nproc_per_node=2 exp/gen_lc0_tc_BT4.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29520 --nproc_per_node=8 exp/gen_lc0_tc_T82.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29530 --nproc_per_node=8 exp/gen_lc0_tc_T82_M14.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29640 --nproc_per_node=8 exp/gen_lc0_tc_2d_T82.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29642 --nproc_per_node=8 exp/gen_lc0_tc_2d_T82_M14.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29620 --nproc_per_node=8 exp/gen_lc0_tc_BT4.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29640 --nproc_per_node=8 exp/gen_lc0_tc_BT4_M14.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29610 --nproc_per_node=8 exp/gen_lc0_tc_2d_BT4.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29680 --nproc_per_node=8 exp/gen_lc0_tc_2d_BT4_M14.py

# gen sae
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29690 --nproc_per_node=8 exp/gen_lc0_sae_BT4.py
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29695 --nproc_per_node=8 exp/gen_lc0_sae_2d_BT4.py

# gen lorsa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29620 --nproc_per_node=8 exp/gen_lc0_lorsa_2d_T82.py
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 torchrun --master_port=29630 --nproc_per_node=6 exp/gen_lc0_lorsa_2d_T82.py

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29620 --nproc_per_node=8 exp/gen_lc0_lorsa_2d_BT4.py


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29690 --nproc_per_node=8 exp/gen_lc0_lorsa_2d_BT4_lichess_puzzle.py
# eval
# eval tc
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=30010 --nproc-per-node=1 exp/eval_lc0_tc.py --layer 14


# 起mongodb
db.fs.chunks.findOne({}, { data: 1 })

export MONGO_URI=$(cat $HOME/mongoip)

# 查看最新的mongourl

mongosh mongodb://10.245.40.143:27017/mechinterp    
# db.datasets.find()
# db.saes.find()
# db.models.find({"name":"searchless-chess-9M-behavioral-cloning"})
# db.models.deleteMany({"name":"searchless-chess-270M"})
# db.analyses.find({"sae_name":"searchless_chess-test-L3"})
# db.analyses.deleteMany({"sae_name":"searchless_chess-test-L8"})
# db.analyses.deleteOne({"sae_name":"searchless_chess-test-L3"})
# db.datasets.deleteOne({ name: 'puzzle_demo' })
# db.features.findOne({"sae_name":"lc0-test-L0-15-master-exp16"})
# db.features.findOne({"sae_name":"searchless_chess-test-L8"})

# 每次新开analyses要注意的
# db.analyses.deleteMany({"sae_name":"lc0-test-L0-15-master-exp16"})
# db.features.deleteMany({"sae_name":"lc0-test-L0-15-master-exp16"})


# uv run lm-saes create dataset puzzle_demo /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/json_data/puzzle_demo_hf/dataset_info.json

# 下面一行应该有问题
# uv run lm-saes create model searchless-chess-270M /inspire/hdd/global_user/hezhengfu-240208120186/models/chess/searchless_chess/dummy_config.json
uv run lm-saes create model searchless-chess-9M-behavioral-cloning /inspire/hdd/global_user/hezhengfu-240208120186/models/chess/searchless_chess/dummy_config_9BC.json
uv run lm-saes create dataset puzzle_demo /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/merge_chess_dedup_fen_only/dataset/dataset_info.json
uv run lm-saes create sae searchless_chess-test-L8 searchless_chess-test /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/results_tmux



uv run lm-saes create model lc0/T82-768x15x24h /inspire/hdd/global_user/hezhengfu-240208120186/models/chess/lc0/dummy_config_lc0_T82-768x15x24h.json
                                              /inspire/hdd/global_user/hezhengfu-240208120186/models/chess/lc0/dummy_config_lc0_T82-768x15x24h copy.json
uv run lm-saes create model lc0/BT4-1024x15x32h /inspire/hdd/global_user/hezhengfu-240208120186/models/chess/leela-BT4/dummy_config_lc0_BT4-1024x15x32h.json
uv run lm-saes create dataset chess_full_puzzle_demo /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/chess_compilation_data/dataset_info.json
uv run lm-saes create dataset master /inspire/hdd/global_user/hezhengfu-240208120186/data/rlin_data/Chess/chess_master_data/dataset_info.json
 
uv run uvicorn server.app:app --host 0.0.0.0 --port 3000 --env-file server/.env


# card_occupy

CUDA_VISIBLE_DEVICES=3,2,1,0 torchrun --master_port=29910 --nproc_per_node=4 exp/card_occupy.py 
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --master_port=29910 --nproc_per_node=4 exp/card_occupy.py 
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=29910 --nproc_per_node=1 exp/card_occupy.py 
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=29910 --nproc_per_node=1 exp/card_occupy.py 


rsync -a --delete /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/activations_lc0/empty \
      /inspire/hdd/global_user/hezhengfu-240208120186/rlin_projects/rlin_projects/chess-SAEs/activations_lc0/clt/


# linear policy head training (distillation)
python exp/06policy_head_distillation/train_stable.py

CUDA_VISIBLE_DEVICES=0 torchrun --master_port=29500 --nproc_per_node=1 exp/card_occupy.py 


# policy lens
python exp/09lens/policy_lens.py
python exp/09lens/visualization/visualize_policy_lens.py

python exp/10selfplay/selfplay.py


# train sae

# analyze sae
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=30000 --nproc-per-node=1 exp/analyze_lc0_tc.py

# train lorsa
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --master-port=29300 exp/train_lorsa.py --lr 1e-4 --layer 1
# done
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29100 exp/train_lorsa.py --lr 1e-4 --layer 5

WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29600 exp/train_lorsa.py --lr 1e-4 --layer 11
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29100 exp/train_lorsa.py --lr 1e-4 --layer 3
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --master-port=29200 exp/train_lorsa.py --lr 1e-4 --layer 0
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --master-port=29300 exp/train_lorsa.py --lr 1e-4 --layer 1
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29320 exp/train_lorsa.py --lr 1e-4 --layer 2
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29350 exp/train_lorsa.py --lr 1e-4 --layer 4

# running
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29260 exp/train_lorsa.py --lr 1e-4 --layer 6
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29600 exp/train_lorsa.py --lr 1e-4 --layer 7
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29700 exp/train_lorsa.py --lr 1e-4 --layer 8
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --master-port=29400 exp/train_lorsa.py --lr 1e-4 --layer 9
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --master-port=29410 exp/train_lorsa.py --lr 1e-4 --layer 10

# 测试lorsa是否完美重构
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 torchrun --nproc-per-node=1 --master-port=29300 exp/train_lorsa_recon_test.py --lr 1e-4 --layer 1


for L in $(seq 0 14); do
  WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=1 \
  torchrun --nproc-per-node=1 --master-port=$((29400+L)) \
  exp/train_lorsa.py --lr 1e-4 --layer $L
done

# todo
WANDB_MODE=offline WANDB_CONSOLE=off CUDA_VISIBLE_DEVICES=0 torchrun --nproc-per-node=1 --master-port=29800 exp/train_lorsa.py --lr 1e-5 --layer 10

# analyse tc
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=30070 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 0
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=30060 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 1


CUDA_VISIBLE_DEVICES=0 torchrun --master_port=30050 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 2
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=30040 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 3
CUDA_VISIBLE_DEVICES=2 torchrun --master_port=30000 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 4
CUDA_VISIBLE_DEVICES=3 torchrun --master_port=30090 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 5


CUDA_VISIBLE_DEVICES=4 torchrun --master_port=30080 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 6
CUDA_VISIBLE_DEVICES=5 torchrun --master_port=30071 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 7
CUDA_VISIBLE_DEVICES=6 torchrun --master_port=30061 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 8
CUDA_VISIBLE_DEVICES=7 torchrun --master_port=30051 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 9

CUDA_VISIBLE_DEVICES=7 torchrun --master_port=30040 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 10
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=30030 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 11
CUDA_VISIBLE_DEVICES=2 torchrun --master_port=30020 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 12
CUDA_VISIBLE_DEVICES=3 torchrun --master_port=30010 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 13
CUDA_VISIBLE_DEVICES=4 torchrun --master_port=30000 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 14



CUDA_VISIBLE_DEVICES=0 torchrun --master_port=30000 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 14
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=30000 --nproc-per-node=1 exp/analyze_lc0_tc.py --layer 0
# analyze lorsa
CUDA_VISIBLE_DEVICES=0 torchrun --master_port=30100 --nproc-per-node=1 exp/analyze_lc0_lorsa.py --layer 5
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=0 torchrun --standalone --max_restarts=3 --master_port=30100 --nproc-per-node=1 exp/analyze_lc0_lorsa.py --layer 7
CUDA_VISIBLE_DEVICES=1 CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --max_restarts=3 --master_port=30000 --nproc-per-node=1 exp/analyze_lc0_lorsa.py --layer 5



# eval lorsa
CUDA_VISIBLE_DEVICES=1 torchrun --master_port=30010 --nproc-per-node=1 exp/eval_lc0_lorsa.py --layer 0


# pytest
RUN_BT4_GPU_TESTS=1 \
  BT4_TC_BASE_PATH="result_BT4/tc/k_30_e_16" \
  BT4_LORSA_BASE_PATH="result_BT4/lorsa/k_30_e_16" \
  BT4_GPU_MAX_FEATURE_NODES=4096 \
  BT4_GPU_VJP_BATCH_SIZE=64 \
  uv run pytest -q -s tests/integration/test_attribution_qk_gpu.py

RUN_BT4_GPU_TESTS=1 \
  BT4_TC_BASE_PATH="result_BT4/tc/k_30_e_16" \
  BT4_LORSA_BASE_PATH="result_BT4/lorsa/k_30_e_16" \
  BT4_GPU_MAX_FEATURE_NODES=4096 \
  BT4_GPU_VJP_BATCH_SIZE=64 \
  uv run pytest -q -s tests/integration/test_attribution_qk_gpu.py


uv run pytest -q -s \
    tests/unit/test_attribution_qk_fast.py::test_graph_influence_promotes_mixed_precision_edges_to_float32