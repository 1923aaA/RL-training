#!/usr/bin/env python
import sys
import os
import torch
import shutil  # <--- 新增这行
import numpy as np
from pathlib import Path
import setproctitle

# 路径设置
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(parent_path)

from onpolicy.config import get_config
# [关键] 导入你写好的 wrapper 和 config
from onpolicy.envs.kill_chain.env_wrappers import make_train_env, make_eval_env
from onpolicy.envs.kill_chain.config import Config as KillChainConfig

#使用tensorboard生成奖励值变化曲线
#tensorboard --logdir="C:\Users\AeroZeng\Desktop\123456\mappo-main_stable\onpolicy\scripts\results\KillChain"
#tensorboard --logdir="C:\Users\AeroZeng\Desktop\123456\mappo-main_comm_no_net\offpolicy\scripts\results\KillChain\mqmix\Debug_RNN"
#tensorboard --logdir="D:\self\sci\kill_chain\mappo-main_comm3.0_plus\onpolicy\scripts\results\KillChain\rmappo\SAMAPPO_Ex4_4_4_PLUS\run1"


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default='kill_chain_v6', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                       0] + "/results")/ all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists(): os.makedirs(str(run_dir))

    exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                     str(folder.name).startswith('run')]
    curr_run = 'run1' if len(exst_run_nums) == 0 else 'run%i' % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists(): os.makedirs(str(run_dir))

    # ================= [新增代码开始] =================
    # 利用前面定义好的 parent_path (项目根目录) 定位 config.py
    config_src = os.path.join(parent_path, "onpolicy", "envs", "kill_chain", "config.py")
    config_dst = run_dir / "config.py"  # run_dir 是 Path 对象，可以直接用 / 拼接

    try:
        shutil.copy(config_src, config_dst)
        print(f"[Info] Config file backed up to: {config_dst}")
    except FileNotFoundError:
        print(f"[Warning] Config file not found at: {config_src}")
    except Exception as e:
        print(f"[Warning] Failed to backup config file: {e}")
    # ================= [新增代码结束] =================

    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 从你的 Config 读取智能体数量
    kc_config = KillChainConfig()
    all_args.num_agents = kc_config.n_agents
    print(f"Detected {all_args.num_agents} agents from KillChain Config.")

    # 使用你的 make_train_env
    envs = make_train_env(all_args.env_name, all_args.seed, all_args.n_rollout_threads, all_args)
    eval_envs = make_eval_env(all_args.env_name, all_args.seed, all_args.n_eval_rollout_threads,
                              all_args) if all_args.use_eval else None

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir
    }

    from onpolicy.runner.shared.kill_chain_runner import KillChainRunner as Runner

    runner = Runner(config)
    runner.run()

    envs.close()
    if eval_envs is not None: eval_envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])