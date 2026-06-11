# gen TC
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29640 --nproc_per_node=8 exp/gen_lc0_tc_2d_T82.py

# gen Lorsa
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun --master_port=29620 --nproc_per_node=8 exp/gen_lc0_lorsa_2d_T82.py