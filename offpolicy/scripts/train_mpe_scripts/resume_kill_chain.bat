@echo off
set env=KillChain
set algo=mqmix
set exp=QMIX_Ex10_10_4FH
set seed=1

set model_path=D:\self\sci\kill_chain\mappo-main_comm3.0\offpolicy\scripts\results\KillChain\mqmix\QMIX_Ex4_4_4FH\run1\models\

set CUDA_VISIBLE_DEVICES=0

python ../train/train_kill_chain.py ^
    --env_name %env% ^
    --algorithm_name %algo% ^
    --experiment_name %exp% ^
    --seed %seed% ^
    --n_training_threads 1 ^
    --n_rollout_threads 8 ^
    --num_env_steps 30000000 ^
    --episode_length 300 ^
    --batch_size 64 ^
    --buffer_size 3000 ^
    --lr 3e-4 ^
    --epsilon_anneal_time 3000000 ^
    --hidden_size 128 ^
    --data_chunk_length 40 ^
    --use_popart ^
    --prev_act_inp ^
    --model_dir %model_path%

pause