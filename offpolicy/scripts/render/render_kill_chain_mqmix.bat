@echo off
set env=KillChain
set algo=mqmix
set exp=render_demo
set seed=1

set model_dir="D:\self\sci\kill_chain\mappo-main_comm3.0\compare4_4_4\QMIX_Ex4_4_4\models\policy_0"

set CUDA_VISIBLE_DEVICES=0
set render_episodes=1000
set ifi=0.01

echo "=================================================="
echo "Rendering KillChain (MQMix)..."
echo "Model Path: %model_dir%"
echo "=================================================="

python "%~dp0render_kill_chain_mqmix.py" ^
    --env_name %env% ^
    --algorithm_name %algo% ^
    --experiment_name %exp% ^
    --seed %seed% ^
    --render_episodes %render_episodes% ^
    --n_training_threads 1 ^
    --n_rollout_threads 1 ^
    --episode_length 150 ^
    --share_policy ^
    --hidden_size 128 ^
    --ifi %ifi% ^
    --model_dir %model_dir%

pause