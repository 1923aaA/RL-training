@echo off
set env=KillChain
set algo=rmappo
set exp=SAMAPPO_SCENARIO_DEBUG_PLUS
set seed=3

set CUDA_VISIBLE_DEVICES=0
set KMP_DUPLICATE_LIB_OK=TRUE

python ../train/train_kill_chain.py ^
    --env_name %env% ^
    --algorithm_name %algo% ^
    --experiment_name %exp% ^
    --seed %seed% ^
    --n_training_threads 1 ^
    --n_rollout_threads 32 ^
    --num_mini_batch 2 ^
    --episode_length 240^
    --num_env_steps 30000000 ^
    --max_grad_norm 0.3 ^
    --ppo_epoch 5 ^
    --gain 0.01 ^
    --lr 1e-4 ^
    --critic_lr 1e-4 ^
    --use_attention ^
    --clip_param 0.1

pause