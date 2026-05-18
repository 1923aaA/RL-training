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

from offpolicy.config import get_config
from offpolicy.envs.kill_chain.KillChain_Env import KillChainEnv
from offpolicy.algorithms.mqmix.algorithm.mQMixPolicy import M_QMixPolicy

# 导入可视化工具
# 这里的 try-except 是为了防止有些环境下没有 vis_utils 报错
try:
    from vis_utils import KillChainVisualizer
except ImportError:
    KillChainVisualizer = None


def parse_args(args, parser):
    parser.add_argument("--scenario_name", type=str, default='kill_chain_v1')
    parser.add_argument("--num_landmarks", type=int, default=3)
    # 默认无头模式，加上 --use_gui 才会显示界面
    parser.add_argument("--use_gui", action='store_true', default=False, help="是否开启Matplotlib图形化")
    parser.set_defaults(render_episodes=1000, ifi=0.1)
    all_args = parser.parse_known_args(args)[0]
    return all_args


def setup_logger(model_dir):
    if not model_dir: return None, None, []
    model_path = Path(model_dir)
    results_dir = model_path.parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = results_dir / f"eval_mqmix_{timestamp}.csv"

    # ================= [修复] 补全所有指标 =================
    metric_keys = [
        # --- 核心成果 ---
        "kill_chain/is_success",  # 是否任务全部清空
        "kill_chain/success_rate",  # 成功率 (杀敌数/任务数)
        "kill_chain/completed_tasks",  # 完成任务数
        "kill_chain/cooperation_count",  # 协作杀敌数
        "kill_chain/total_reward",  # 总奖励

        # --- 杀伤链效率 ---
        "kill_chain/avg_pk",  # 平均杀伤概率
        "kill_chain/avg_time_eff",  # 平均时间效率 (Env计算的)
        "kill_chain/paper_temporal_efficiency",  # 论文定义的时间效率 (1/time)

        # --- 分角色奖励 (S/D/I) ---
        "kill_chain/reward_S",
        "kill_chain/reward_D",
        "kill_chain/reward_I",

        # --- 网络与性能指标 ---
        "metrics/avg_hops",  # 平均跳数
        "metrics/avg_ttc",  # 平均闭环耗时 (Time To Close)
        "metrics/energy_per_kill",  # 单次击杀能耗
        "metrics/avg_buffer_load",  # 平均 Buffer 负载
        "metrics/msg_throughput"  # 消息吞吐量
    ]
    # =======================================================

    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["episode"] + metric_keys)
    print(f"[Log] 数据将保存至: {csv_path}")
    return csv_path, metric_keys


def save_episode_data(csv_path, metric_keys, episode_idx, info):
    if not csv_path or not info: return
    row_data = [episode_idx + 1]
    for key in metric_keys:
        # 使用 get(key, 0) 确保即使 info 缺失某些键也不会报错
        val = info.get(key, 0.0)
        if isinstance(val, (float, np.floating)): val = round(val, 4)
        row_data.append(val)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def print_battle_report(env_instance):
    print(f"\n{'=' * 30} 本局战报 {'=' * 30}")
    tasks = env_instance.tasks
    completed = 0
    for task in tasks:
        status = task.stage.name
        color_code = "\033[92m" if status == 'COMPLETED' else "\033[91m"
        reset_code = "\033[0m"
        print(f"Task [{task.task_id}]: {color_code}{status}{reset_code}")
        if status == 'COMPLETED': completed += 1
    print(f"Summary: {completed}/{len(tasks)}")


# 辅助函数：将 One-Hot 动作转为环境需要的索引
def onehot_to_indices(onehot_actions, act_space):
    indices_list = []
    if hasattr(act_space, 'nvec'):  # MultiDiscrete
        current_idx = 0
        for dim in act_space.nvec:
            part = onehot_actions[:, :, current_idx: current_idx + dim]
            idx = np.argmax(part, axis=-1)
            indices_list.append(idx)
            current_idx += dim
        return np.stack(indices_list, axis=-1)
    else:  # Discrete
        return np.argmax(onehot_actions, axis=-1)


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # 强制覆盖部分参数以适应推理
    all_args.use_eval = False
    all_args.cuda = False
    all_args.n_rollout_threads = 1
    all_args.n_eval_rollout_threads = 1

    if all_args.model_dir is None or not os.path.exists(str(all_args.model_dir)):
        print(f"错误: 模型路径无效，请通过 --model_dir 指定")
        return

    csv_path, metric_keys = setup_logger(all_args.model_dir)

    # 1. 初始化环境
    env = KillChainEnv(all_args)
    env.seed(all_args.seed)

    # =======================================================
    # [关键修复] 将环境的 agent 数量注入到 args 中
    # MQMixPolicy 初始化时需要访问 args.num_agents
    # =======================================================
    all_args.num_agents = env.n_agents

    # 2. 初始化可视化
    visualizer = None
    if all_args.use_gui:
        if KillChainVisualizer is None:
            print("[Warning] vis_utils 未找到，无法启动 GUI")
        else:
            space_size = getattr(env.cfg, 'space_size', 100.0)
            visualizer = KillChainVisualizer(env, space_size=space_size, refresh_interval=all_args.ifi)
    else:
        print("[Info] 运行模式: 无界面极速测试 (Headless Mode)")

    # 3. 初始化 Policy
    policy_config = {"args": all_args}
    policy_info = {
        'act_space': env.action_space[0],
        'obs_space': env.observation_space[0],
        'share_obs_space': env.share_observation_space[0]
    }

    # 现在 args.num_agents 已经有了，初始化不会报错了
    policy = M_QMixPolicy(policy_config, policy_info)

    # 4. 加载模型权重
    model_path = os.path.join(all_args.model_dir, "q_network.pt")
    if not os.path.exists(model_path):
        if os.path.isfile(all_args.model_dir):
            model_path = all_args.model_dir
        else:
            # 尝试加载可能存在的其他命名
            model_path = os.path.join(all_args.model_dir, "actor.pt")

    print(f"[Load] Loading model from: {model_path}")
    # map_location='cpu' 确保在无 GPU 环境下也能跑
    state_dict = torch.load(model_path, map_location=torch.device("cpu"))
    policy.q_network.load_state_dict(state_dict)
    policy.prep_rollout()

    print(f"[Start] 开始评估 {all_args.render_episodes} 个回合 (Algorithm: MQMix)...")

    for episode in range(all_args.render_episodes):
        print(f"--- Episode {episode + 1} Start ---")
        obs, share_obs, available_actions = env.reset()

        obs = np.expand_dims(obs, 0)
        available_actions = np.expand_dims(available_actions, 0)

        num_agents = env.n_agents
        rnn_states = np.zeros((1, num_agents, all_args.hidden_size), dtype=np.float32)

        # MQMix 需要上一时刻的动作作为输入
        act_dim = policy.act_dim
        prev_onehot_actions = np.zeros((1, num_agents, act_dim), dtype=np.float32)

        step = 0
        while True:
            with torch.no_grad():
                actions_onehot, rnn_states = policy.get_actions(
                    obs,
                    prev_onehot_actions,
                    rnn_states,
                    available_actions,
                    explore=False
                )

                actions_onehot = actions_onehot.detach().cpu().numpy()
                rnn_states = rnn_states.detach().cpu().numpy()

            actions_env = onehot_to_indices(actions_onehot, env.action_space[0])
            prev_onehot_actions = actions_onehot

            obs, share_obs, rewards, dones, infos, available_actions = env.step(actions_env[0])

            if visualizer:
                visualizer.render(step)

            if np.all(dones):
                print_battle_report(env)
                if infos:
                    # 获取 info 字典 (infos 是列表，取第一个即可)
                    info_dict = infos[0]
                    save_episode_data(csv_path, metric_keys, episode, info_dict)

                    # 打印一些核心指标到控制台确认
                    succ = info_dict.get('kill_chain/is_success', 0)
                    pk = info_dict.get('kill_chain/avg_pk', 0)
                    print(f"[Log] Episode {episode + 1} metrics saved. (Success: {succ}, AvgPk: {pk:.3f})")

                time.sleep(0.1)
                break

            obs = np.expand_dims(obs, 0)
            available_actions = np.expand_dims(available_actions, 0)
            step += 1

    if visualizer:
        visualizer.close()

    print(f"所有测试结束，数据已保存至: {csv_path}")


if __name__ == "__main__":
    main(sys.argv[1:])