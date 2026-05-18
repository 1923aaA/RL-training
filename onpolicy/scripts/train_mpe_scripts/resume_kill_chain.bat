@echo off
set env=KillChain
set algo=rmappo
set exp=SA-MAPPO_SCENARIO_DEBUG_PLUS
set seed=2

set model_path="D:\self\sci\kill_chain\mappo-main_comm3.0_plus_2026_5\onpolicy\scripts\results\KillChain\rmappo\SAMAPPO_SCENARIO_DEBUG_PLUS\run4\models"

set CUDA_VISIBLE_DEVICES=0

python ../train/train_kill_chain.py ^
    --env_name %env% ^
    --algorithm_name %algo% ^
    --experiment_name %exp% ^
    --seed %seed% ^
    --n_training_threads 1 ^
    --n_rollout_threads 16 ^
    --num_mini_batch 1 ^
    --episode_length 450 ^
    --num_env_steps 30000000 ^
    --ppo_epoch 15 ^
    --gain 0.01 ^
    --lr 3e-4 ^
    --critic_lr 3e-4 ^
    --share_policy ^
    --use_attention^
    --model_dir %model_path%

pause