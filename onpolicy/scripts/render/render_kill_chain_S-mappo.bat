@echo off
set env=KillChain
set algo=rmappo
:: r_a2c   rmappo   qmix
set exp=render_demo
set seed=5

set model_dir="D:\self\sci\kill_chain\mappo-main_comm3.0_plus\onpolicy\scripts\results\KillChain\rmappo\SAMAPPO_Ex10_10_4_PLUS\run1\models"

set CUDA_VISIBLE_DEVICES=0
set render_episodes=1
set ifi=0.1

set attention_flag=--use_attention
:: set attention_flag=

echo "=================================================="
echo "Rendering KillChain..."
echo "Model: %model_dir%"
echo "Attention Mode: %attention_flag%"
echo "=================================================="

python ../render/render_kill_chain.py ^
    --env_name %env% ^
    --algorithm_name %algo% ^
    --experiment_name %exp% ^
    --seed %seed% ^
    --render_episodes %render_episodes% ^
    --n_training_threads 1 ^
    --n_rollout_threads 1 ^
    --episode_length 300 ^
    --share_policy ^
    --ifi %ifi% ^
    --model_dir %model_dir% ^
    --eval_n_target 12 ^
    --share_policy ^
    %attention_flag% ^

pause