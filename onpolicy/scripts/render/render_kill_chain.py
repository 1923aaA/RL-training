#!/usr/bin/env python
import sys
import os
import time
import torch
import numpy as np
import csv
import datetime
from pathlib import Path

# 添加路径
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(parent_path)

from onpolicy.config import get_config
from onpolicy.envs.kill_chain.KillChain_Env import KillChainEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy

# 导入可视化工具
from vis_utils import KillChainVisualizer


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default='kill_chain_v1')
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--save_frames", action='store_true', help="是否保存关键帧图片")
    parser.add_argument("--use_gui", action='store_true', default=False, help="是否开启 Matplotlib 图形化")
    parser.add_argument("--eval_n_target", type=int, default=-1, help="固定可视化目标数量，-1 表示使用配置默认")

    parser.set_defaults(render_episodes=10, ifi=0.1)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def setup_logger(model_dir):
    """
    配置 CSV 记录器
    保存位置: .../run1/models/ -> .../run1/results/eval_metrics_xxx.csv
    """
    if not model_dir:
        return None, None

    model_path = Path(model_dir)
    results_dir = model_path.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"eval_metrics_{timestamp}.csv"

    metric_keys = [
        "kill_chain/current_n_target",
        "kill_chain/total_reward",
        "kill_chain/reward_per_target",
        "kill_chain/completed_tasks",
        "kill_chain/completion_ratio",
        "kill_chain/clear_rate",
        "kill_chain/success_rate",
        "kill_chain/reward_S",
        "kill_chain/reward_D",
        "kill_chain/reward_I",
        "kill_chain/avg_pk",
        "kill_chain/avg_time_eff",
        "kill_chain/paper_temporal_efficiency",
        "metrics/avg_hops",
        "metrics/avg_ttc",
        "metrics/energy_per_kill",
        "metrics/avg_buffer_load",
        "metrics/msg_throughput"
    ]

    headers = ["episode"] + metric_keys

    print(f"[Log] 数据将保存至: {csv_path}")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    return csv_path, metric_keys


def save_episode_data(csv_path, metric_keys, episode_idx, info):
    """将单回合数据追加写入 CSV"""
    if not csv_path or not info:
        return

    row_data = [episode_idx + 1]

    for key in metric_keys:
        val = info.get(key, 0)
        if isinstance(val, (float, np.floating)):
            val = round(float(val), 4)
        row_data.append(val)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def print_battle_report(env_instance):
    """控制台打印简报"""
    print(f"\n{'=' * 30} 本局战报 {'=' * 30}")
    tasks = env_instance.tasks
    completed = 0

    for task in tasks:
        status = task.stage.name
        color_code = "\033[92m" if status == 'COMPLETED' else "\033[91m"
        reset_code = "\033[0m"

        chain_info = ""
        if status == 'COMPLETED' and hasattr(task, 'final_kill_prob'):
            try:
                chain_info = f" | Pk: {float(task.final_kill_prob):.2f}"
            except Exception:
                chain_info = ""

        print(f"Task [{task.task_id}]: {color_code}{status}{reset_code}{chain_info}")
        if status == 'COMPLETED':
            completed += 1

    print(f"Summary: {completed}/{len(tasks)}")


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # 渲染/评估统一强制 CPU + 单环境
    all_args.use_eval = False
    all_args.cuda = False
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1

    if all_args.model_dir is None or not os.path.exists(str(all_args.model_dir)):
        print("错误: 模型路径无效，请通过 --model_dir 指定")
        return

    csv_path, metric_keys = setup_logger(all_args.model_dir)

    # 关键：渲染时应使用评估模式
    env = KillChainEnv(all_args, rank=0, eval_mode=True)
    if getattr(all_args, "eval_n_target", -1) > 0:
        env.cfg.eval_n_target_list = [int(all_args.eval_n_target)]
    env.seed(all_args.seed)

    visualizer = None
    if all_args.use_gui:
        space_size = getattr(env.cfg, 'space_size_x', 100.0)
        visualizer = KillChainVisualizer(
            env,
            space_size=space_size,
            refresh_interval=all_args.ifi,
            save_frames=all_args.save_frames
        )
    else:
        print("[Info] 运行模式: 无界面极速测试 (Headless Mode)")

    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]
    num_agents = env.n_agents

    policy = R_MAPPOPolicy(
        all_args,
        obs_space,
        share_obs_space,
        act_space,
        device=torch.device("cpu")
    )
    policy.restore(all_args.model_dir)

    print(f"[Start] 开始评估 {all_args.render_episodes} 个回合...")
    print(f"[Info] use_attention = {getattr(all_args, 'use_attention', False)}")
    print(f"[Info] actor base = {type(policy.actor.base).__name__}")

    for episode in range(all_args.render_episodes):
        print(f"--- Episode {episode + 1} Start ---")

        obs, share_obs, available_actions = env.reset()

        obs = np.expand_dims(obs, 0)
        share_obs = np.expand_dims(share_obs, 0)
        available_actions = np.expand_dims(available_actions, 0)

        rnn_states = np.zeros(
            (1, num_agents, all_args.recurrent_N, all_args.hidden_size),
            dtype=np.float32
        )
        # 关键：episode 起始 mask 设为 0，清空 RNN 记忆
        masks = np.zeros((1, num_agents, 1), dtype=np.float32)

        step = 0
        while True:
            with torch.no_grad():
                actions, rnn_states_actor = policy.act(
                    obs[0],
                    rnn_states[0],
                    masks[0],
                    available_actions[0],
                    deterministic=True
                )

            actions_env = actions.detach().cpu().numpy()
            rnn_states_actor = rnn_states_actor.detach().cpu().numpy()

            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions_env)

            if visualizer:
                visualizer.render(step)

            if np.all(dones):
                print_battle_report(env)

                if infos and len(infos) > 0:
                    info_dict = infos[0]
                    save_episode_data(csv_path, metric_keys, episode, info_dict)
                    print(f"[Log] Episode {episode + 1} metrics saved.")
                    print(
                        f"[Result] N={info_dict.get('kill_chain/current_n_target', 0)}, "
                        f"completion_ratio={info_dict.get('kill_chain/completion_ratio', 0):.3f}, "
                        f"clear_rate={info_dict.get('kill_chain/clear_rate', 0):.3f}, "
                        f"reward_per_target={info_dict.get('kill_chain/reward_per_target', 0):.3f}"
                    )

                time.sleep(0.1)
                break

            obs = np.expand_dims(obs, 0)
            share_obs = np.expand_dims(share_obs, 0)
            available_actions = np.expand_dims(available_actions, 0)
            rnn_states[0] = rnn_states_actor

            # 非首步时保持记忆
            masks[0] = 1.0

            step += 1

    if visualizer:
        visualizer.close()

    print(f"所有测试结束，数据已保存至: {csv_path}")


if __name__ == "__main__":
    main(sys.argv[1:])