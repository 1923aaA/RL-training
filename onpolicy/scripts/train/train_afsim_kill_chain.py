#!/usr/bin/env python
"""
AFSIM KillChain 训练启动脚本。

用法示例:
  # 真实 AFSIM 模式 (需要 pydds + AFSIM 场景运行):
  python train_afsim_kill_chain.py --env_name AFSIMKillChain --algorithm_name rmappo \\
      --experiment_name afsim_v1 --num_env_steps 2000000 --episode_length 200

  # Mock 模式 (本地测框架, 不需要 AFSIM):
  python train_afsim_kill_chain.py --env_name AFSIMKillChain --algorithm_name rmappo \\
      --experiment_name mock_test --num_env_steps 5000 --episode_length 50 \\
      --afsim_mock_mode
"""
import sys
import os
import shutil
import torch
import numpy as np
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    force=True,
)
from pathlib import Path
import setproctitle

# 路径设置
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(parent_path)

from onpolicy.config import get_config

from onpolicy.envs.afsim_kill_chain.env_wrappers import make_train_env, make_eval_env
from onpolicy.envs.afsim_kill_chain.config import Config as AFSIMKillChainConfig


def parse_args(args, parser):
    """AFSIM KillChain 专有参数"""
    parser.add_argument("--scenario_name", type=str, default='afsim_kill_chain_v1',
                        help="场景标识")
    parser.add_argument("--afsim_mock_mode", action="store_true", default=False,
                        help="启用 mock 模式, 不真正连 DDS (本地调试用)")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # AFSIM 约束: 单实例不能并行
    if all_args.n_rollout_threads != 1:
        print(f"[Warning] AFSIM 单实例, 强制 n_rollout_threads 从 "
              f"{all_args.n_rollout_threads} 改为 1")
        all_args.n_rollout_threads = 1
    if all_args.use_eval and all_args.n_eval_rollout_threads != 1:
        print(f"[Warning] AFSIM 单实例, 强制 n_eval_rollout_threads 从 "
              f"{all_args.n_eval_rollout_threads} 改为 1")
        all_args.n_eval_rollout_threads = 1

    # 环境名默认覆盖
    if all_args.env_name != "AFSIMKillChain":
        print(f"[Info] 建议 env_name 设为 'AFSIMKillChain' (当前: '{all_args.env_name}')")

    # 设备
    if all_args.cuda and torch.cuda.is_available():
        print("[Info] 使用 GPU")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
    else:
        print("[Info] 使用 CPU")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # 运行目录
    run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0]
                   + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 自动编号 (runN)
    existing_run_nums = [
        int(str(folder.name).split("run")[1])
        for folder in run_dir.iterdir() if str(folder.name).startswith("run")
    ]
    curr_run = "run1" if not existing_run_nums else f"run{max(existing_run_nums) + 1}"
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # 备份 AFSIM config 到 run 目录
    config_src = os.path.join(parent_path, "onpolicy", "envs", "afsim_kill_chain", "config.py")
    config_dst = run_dir / "afsim_config_backup.py"
    try:
        shutil.copy(config_src, config_dst)
        print(f"[Info] 已备份 AFSIM config 到 {config_dst}")
    except Exception as e:
        print(f"[Warning] 备份 config 失败: {e}")

    # 进程名
    setproctitle.setproctitle(
        f"{all_args.algorithm_name}-{all_args.env_name}-{all_args.experiment_name}@{all_args.user_name}"
    )

    # 随机种子
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 从 AFSIM Config 读 agent 数量 (被场景选择决定)
    afsim_cfg = AFSIMKillChainConfig()
    all_args.num_agents = afsim_cfg.n_agents
    print(f"[Info] AFSIM KillChain 场景: n_agents={all_args.num_agents}, "
          f"n_target={afsim_cfg.n_target}, world={afsim_cfg.world_bounds_km} km")
    print(f"[Info] AFSIM mock_mode={all_args.afsim_mock_mode}, "
          f"step_wait={afsim_cfg.afsim.step_wait_sec}s")

    # 估算训练时长 (AFSIM real-time 下很重要!)
    expected_seconds = (all_args.num_env_steps / all_args.episode_length) * \
                       all_args.episode_length * afsim_cfg.afsim.step_wait_sec
    print(f"[Info] 预计训练墙上时间 (仅 AFSIM 推进): "
          f"{expected_seconds:.0f} 秒 ≈ {expected_seconds / 3600:.1f} 小时 "
          f"(不含模型前向/反向传播和 reset 时间)")

    # 创建环境
    envs = make_train_env(all_args.env_name, all_args.seed, all_args.n_rollout_threads, all_args)
    eval_envs = (make_eval_env(all_args.env_name, all_args.seed,
                               all_args.n_eval_rollout_threads, all_args)
                 if all_args.use_eval else None)

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": all_args.num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # =========================================================================
    # [Fine-tune support] 如果指定了 --model_dir, 启动时打印明显的提示
    # 让你看到 pretrained 是否被正确识别 (base_runner.restore() 会自动加载)
    # =========================================================================
    if all_args.model_dir is not None and str(all_args.model_dir).strip() != "":
        actor_path = os.path.join(str(all_args.model_dir), "actor.pt")
        critic_path = os.path.join(str(all_args.model_dir), "critic.pt")
        vnorm_path = os.path.join(str(all_args.model_dir), "vnorm.pt")
        print("=" * 80)
        print("[Fine-tune Mode] Loading pretrained model from:")
        print(f"  {all_args.model_dir}")
        print(f"  actor.pt:  {'OK' if os.path.exists(actor_path) else 'MISSING'}")
        print(f"  critic.pt: {'OK' if os.path.exists(critic_path) else 'MISSING'}")
        print(f"  vnorm.pt:  {'OK' if os.path.exists(vnorm_path) else 'MISSING (not required)'}")
        print(f"  lr={all_args.lr}, clip_param={all_args.clip_param}, entropy_coef={all_args.entropy_coef}")
        print(f"  ppo_epoch={all_args.ppo_epoch}, save_interval={all_args.save_interval}")
        print("=" * 80)
        if not os.path.exists(actor_path):
            raise FileNotFoundError(f"actor.pt not found at: {actor_path}")
    else:
        print("=" * 80)
        print("[Train from Scratch] No pretrained model_dir provided, training from random init")
        print("=" * 80)

    # Runner
    from onpolicy.runner.shared.afsim_kill_chain_runner import AFSIMKillChainRunner as Runner
    runner = Runner(config)

    try:
        runner.run()
    finally:
        envs.close()
        if eval_envs is not None:
            eval_envs.close()


if __name__ == "__main__":
    main(sys.argv[1:])