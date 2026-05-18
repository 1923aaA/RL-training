#!/usr/bin/env python
"""
AFSIM KillChain 推理脚本 (轻量适配层版).

用法:
  python render_afsim_kill_chain.py \\
      --env_name AFSIMKillChain \\
      --algorithm_name rmappo \\
      --experiment_name afsim_inference \\
      --model_dir D:\\Code\\mappo-main_comm3.0_plus\\onpolicy\\scripts\\results\\KillChain\\rmappo\\SAMAPPO_Ex10_10_4_PLUS\\run1\\models \\
      --render_episodes 5 \\
      --episode_length 200 \\
      --use_eval

前置条件:
  1. AFSIM Warlock 已启动且场景加载完毕
  2. KillChain 模型 (actor.pt + critic.pt + 可选 vnorm.pt) 已训好放在 model_dir
  3. cfg 已切到 SCENARIO_MEDIUM, base_strike_range=1.5 (与训练时一致)

设计要点:
  - 用 R_MAPPOPolicy.act(deterministic=True) 推理, 不更新参数
  - 单 episode 跑完后 env.reset() 自动发 SimReset 让 AFSIM 重摆平台
  - 每 episode 末尾打印战报 + AFSIM Bridge 收发统计

注意:
  - env 用的是 AFSIMKillChainEnv (轻量适配层), 动作空间 [3, 5, 7] 与训练时对齐
  - AFSIM 仅作物理引擎: 位置同步 + 武器命令; 5001/5003 不消费
  - Python 内部多跳 message_queue 保留, 这是 RL 决策的实际驱动力
"""
import sys
import os
import time
import csv
import datetime
from pathlib import Path

import torch
import numpy as np

# 路径设置
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(parent_path)

from onpolicy.config import get_config
from onpolicy.envs.afsim_kill_chain.AFSIMKillChainEnv import AFSIMKillChainEnv
from onpolicy.algorithms.r_mappo.algorithm.rMAPPOPolicy import R_MAPPOPolicy


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default='afsim_kill_chain_inference')
    parser.add_argument("--afsim_mock_mode", action="store_true", default=False,
                        help="启用 mock 模式, 不真正连 DDS (本地调试)")
    parser.add_argument("--save_inference_csv", action="store_true", default=True,
                        help="把每 episode 的指标保存到 CSV")
    parser.add_argument("--eval_n_target", type=int, default=-1,
                        help="固定目标数量, -1 用 cfg 默认")
    parser.set_defaults(render_episodes=5)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def setup_logger(model_dir):
    """配置 CSV 记录器"""
    if not model_dir:
        return None, None

    model_path = Path(model_dir)
    results_dir = model_path.parent / "afsim_inference_results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"afsim_eval_{timestamp}.csv"

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
    ]

    headers = ["episode", "wall_time_sec"] + metric_keys

    print(f"[Log] CSV 保存至: {csv_path}")

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)

    return csv_path, metric_keys


def save_episode_data(csv_path, metric_keys, episode_idx, info, wall_time):
    """追加单 episode 数据到 CSV"""
    if not csv_path or not info:
        return

    row_data = [episode_idx + 1, round(wall_time, 2)]
    for key in metric_keys:
        val = info.get(key, 0)
        if isinstance(val, (float, np.floating)):
            val = round(float(val), 4)
        row_data.append(val)

    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def print_battle_report(env_instance, episode_idx, wall_time):
    """控制台战报"""
    print(f"\n{'=' * 30} Episode {episode_idx + 1} 战报 {'=' * 30}")
    print(f"  墙钟耗时: {wall_time:.1f}s")

    tasks = env_instance.tasks
    completed = 0
    failed = 0
    pending = 0

    for task in tasks:
        status = task.stage.name
        color_code = ""
        if status == 'COMPLETED':
            color_code = "\033[92m"
            completed += 1
        elif status == 'FAILED':
            color_code = "\033[91m"
            failed += 1
        else:
            pending += 1
        reset_code = "\033[0m"

        chain_info = ""
        if status == 'COMPLETED' and hasattr(task, 'final_kill_prob'):
            try:
                chain_info = f" | Pk: {float(task.final_kill_prob):.2f}"
            except Exception:
                chain_info = ""

        print(f"  Task [{task.task_id}]: {color_code}{status}{reset_code}{chain_info}")

    print(f"  Summary: COMPLETED={completed} | FAILED={failed} | "
          f"OTHER={pending} | TOTAL={len(tasks)}")

    # AFSIM Bridge 统计
    if hasattr(env_instance, "bridge"):
        try:
            stats = env_instance.bridge.get_stats()
            print(f"  Bridge: platform_added={stats.get('platform_added_recv', 0)}  "
                  f"move_update={stats.get('move_update_recv', 0)}  "
                  f"strike_evt={stats.get('strike_event_recv', 0)}  "
                  f"fire_sent={stats.get('fire_weapon_sent', 0)}")
        except Exception as e:
            print(f"  [Bridge stats err] {e}")


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # 推理强制 CPU + 单环境
    all_args.use_render = True   # base_runner.restore 会区分
    all_args.cuda = False
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1

    if all_args.model_dir is None or not os.path.exists(str(all_args.model_dir)):
        print(f"错误: 模型路径无效, 请通过 --model_dir 指定")
        print(f"示例: --model_dir D:\\...\\KillChain\\rmappo\\SAMAPPO_Ex10_10_4_PLUS\\run1\\models")
        return

    # actor.pt 必须存在 (critic 推理时不一定要)
    actor_path = os.path.join(str(all_args.model_dir), "actor.pt")
    if not os.path.exists(actor_path):
        print(f"错误: 找不到 actor.pt: {actor_path}")
        return
    print(f"[Info] 加载模型: {actor_path}")

    csv_path, metric_keys = (None, None)
    if all_args.save_inference_csv:
        csv_path, metric_keys = setup_logger(all_args.model_dir)

    # ===== 创建 AFSIM 环境 =====
    print(f"[Info] 创建 AFSIMKillChainEnv (mock_mode={all_args.afsim_mock_mode})...")
    env = AFSIMKillChainEnv(all_args, rank=0, eval_mode=True)

    # 如果指定了固定目标数量, 覆盖 eval_n_target_list
    if getattr(all_args, "eval_n_target", -1) > 0:
        env.cfg.eval_n_target_list = [int(all_args.eval_n_target)]
        print(f"[Info] 固定 eval_n_target = {all_args.eval_n_target}")

    env.seed(all_args.seed)

    # ===== 创建 Policy 并加载训练好的权重 =====
    obs_space = env.observation_space[0]
    share_obs_space = env.share_observation_space[0]
    act_space = env.action_space[0]
    num_agents = env.n_agents

    print(f"[Info] obs_dim={obs_space.shape[0]}, "
          f"share_obs_dim={share_obs_space.shape[0]}, "
          f"action_space={act_space}, n_agents={num_agents}")

    policy = R_MAPPOPolicy(
        all_args,
        obs_space,
        share_obs_space,
        act_space,
        device=torch.device("cpu"),
    )
    policy.restore(str(all_args.model_dir))
    print(f"[Info] Policy 加载成功. actor base = {type(policy.actor.base).__name__}")
    print(f"[Info] use_attention = {getattr(all_args, 'use_attention', False)}")

    # ===== 推理循环 =====
    print(f"\n[Start] 开始 AFSIM 推理 {all_args.render_episodes} 个 episode...")

    try:
        for episode in range(all_args.render_episodes):
            print(f"\n--- Episode {episode + 1} 开始 ---")
            ep_start = time.time()

            obs, share_obs, available_actions = env.reset()

            obs = np.expand_dims(obs, 0)
            share_obs = np.expand_dims(share_obs, 0)
            available_actions = np.expand_dims(available_actions, 0)

            rnn_states = np.zeros(
                (1, num_agents, all_args.recurrent_N, all_args.hidden_size),
                dtype=np.float32,
            )
            # episode 起始: mask=0 清空 RNN 记忆
            masks = np.zeros((1, num_agents, 1), dtype=np.float32)

            step = 0
            while True:
                with torch.no_grad():
                    actions, rnn_states_actor = policy.act(
                        obs[0],
                        rnn_states[0],
                        masks[0],
                        available_actions[0],
                        deterministic=True,
                    )

                actions_env = actions.detach().cpu().numpy()
                rnn_states_actor = rnn_states_actor.detach().cpu().numpy()

                obs, share_obs, rewards, dones, infos, available_actions = env.step(actions_env)

                if np.all(dones):
                    wall_time = time.time() - ep_start
                    print_battle_report(env, episode, wall_time)

                    if infos and len(infos) > 0:
                        info_dict = infos[0] if isinstance(infos, list) else infos
                        save_episode_data(csv_path, metric_keys, episode, info_dict, wall_time)
                        print(
                            f"[Result] N={info_dict.get('kill_chain/current_n_target', 0)}, "
                            f"completion_ratio={info_dict.get('kill_chain/completion_ratio', 0):.3f}, "
                            f"clear_rate={info_dict.get('kill_chain/clear_rate', 0):.3f}, "
                            f"reward_per_target={info_dict.get('kill_chain/reward_per_target', 0):.3f}"
                        )
                    break

                # 处理 obs/avail_actions shape (后续步)
                obs = np.expand_dims(obs, 0)
                share_obs = np.expand_dims(share_obs, 0)
                available_actions = np.expand_dims(available_actions, 0)
                rnn_states[0] = rnn_states_actor

                # 非首步: mask=1 保持记忆
                masks[0] = 1.0

                step += 1

                # 防止 RL 卡死: max_steps 兜底
                if step >= all_args.episode_length * 2:
                    print(f"[Warning] step={step} 超过 episode_length 2 倍, 强制结束")
                    break
    finally:
        # 保证 env.close 把 bridge 关掉
        env.close()

    if csv_path:
        print(f"\n[Done] 所有结果已保存: {csv_path}")
    print("=" * 80)
    print(f"AFSIM 推理结束. 共 {all_args.render_episodes} episodes")


if __name__ == "__main__":
    main(sys.argv[1:])