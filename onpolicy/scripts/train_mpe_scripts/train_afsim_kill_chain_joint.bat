@echo off
set env=AFSIMKillChain
set algo=rmappo
set exp=AFSIM_JOINT_FIXED_TARGETS
set seed=3

set CUDA_VISIBLE_DEVICES=0
set KMP_DUPLICATE_LIB_OK=TRUE

set MODEL_DIR=D:\Project\20260309ZWY\SAMAPPO_SCENARIO_DEBUG_PLUS\run4\models

set PYTHONUNBUFFERED=1
python -u ../train/train_afsim_kill_chain.py ^
    --env_name %env% ^
    --algorithm_name %algo% ^
    --experiment_name %exp% ^
    --seed %seed% ^
    --n_training_threads 1 ^
    --n_rollout_threads 1 ^
    --num_mini_batch 1 ^
    --episode_length 500 ^
    --num_env_steps 2000000 ^
    --max_grad_norm 0.3 ^
    --ppo_epoch 1 ^
    --gain 0.01 ^
    --lr 1e-4 ^
    --critic_lr 1e-4 ^
    --use_attention ^
    --clip_param 0.1 ^
    --model_dir %MODEL_DIR%

pause