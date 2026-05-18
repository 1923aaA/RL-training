@echo off
set env=KillChain
set algo=rmappo
set exp=MAPPO_Ex4_4_4FH
set seed=1

set CUDA_VISIBLE_DEVICES=0

python ../train/train_kill_chain.py ^
    --env_name %env% ^
    --algorithm_name %algo% ^
    --experiment_name %exp% ^
    --seed %seed% ^
    --n_training_threads 1 ^
    --n_rollout_threads 16 ^
    --num_mini_batch 1 ^
    --episode_length 150 ^
    --num_env_steps 20000000 ^
    --max_grad_norm 0.3 ^
    --ppo_epoch 15 ^
    --gain 0.01 ^
    --lr 3e-4 ^
    --critic_lr 3e-4 ^
    --share_policy
pause