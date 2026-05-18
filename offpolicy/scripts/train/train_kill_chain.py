import sys
import os
import wandb
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch
import shutil

# 1. 路径修复
curr_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(curr_path)))
sys.path.append(parent_path)

from offpolicy.config import get_config
from offpolicy.utils.util import get_cent_act_dim, get_dim_from_space
from offpolicy.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
#tensorboard --logdir="C:\Users\AeroZeng\Desktop\123456\mappo-main_comm_no_net\offpolicy\scripts\results\KillChain\rmaddpg"
# 2. 导入自定义 Wrapper
try:
    from offpolicy.envs.kill_chain.wrappers import KillChainOffPolicyWrapper
except ImportError as e:
    print(f"[Error] Import failed: {e}")
    sys.exit(1)


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = KillChainOffPolicyWrapper(all_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = KillChainOffPolicyWrapper(all_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str, default='KillChain', help="Which scenario to run on")
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument('--num_agents', type=int, default=2, help="number of agents")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    # 强制设置环境名称
    all_args.env_name = "KillChain"


    # CUDA 设置
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    base_run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[
                            0] + "/results") / all_args.env_name / all_args.algorithm_name / all_args.experiment_name

    if not base_run_dir.exists():
        os.makedirs(str(base_run_dir))

    curr_run = 1
    while True:
        run_dir = base_run_dir / f"run{curr_run}"
        if not run_dir.exists():
            os.makedirs(str(run_dir))
            print(f"Created log directory: {run_dir}")
            try:
                # 备份 config.py
                # 注意: parent_path 已经在文件顶部定义，指向项目根目录
                config_src = os.path.join(parent_path, "offpolicy", "envs", "kill_chain", "config.py")
                config_dst = os.path.join(str(run_dir), "kill_chain_config_backup.py")

                if os.path.exists(config_src):
                    shutil.copy(config_src, config_dst)
                    print(f"Backup config saved to: {config_dst}")
                else:
                    print(f"[Warning] Config source not found at: {config_src}")
            except Exception as e:
                print(f"[Warning] Failed to backup config: {e}")
            break
        curr_run += 1

    # WandB 初始化
    if all_args.use_wandb:
        run = wandb.init(config=all_args,
                         project=all_args.env_name,
                         entity=all_args.user_name,
                         notes=socket.gethostname(),
                         name=str(all_args.algorithm_name) + "_" +
                              str(all_args.experiment_name) +
                              "_seed" + str(all_args.seed),
                         group=all_args.scenario_name,
                         dir=str(run_dir),
                         job_type="training",
                         reinit=True)

    # 设置进程名称
    setproctitle.setproctitle(
        str(all_args.algorithm_name) + "-" + str(all_args.env_name) + "-" + str(all_args.experiment_name) + "@" + str(
            all_args.user_name))

    # 设置随机种子
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # 创建环境
    env = make_train_env(all_args)
    eval_env = make_eval_env(all_args) if all_args.use_eval else None

    # [重要] 获取智能体数量，使用 num_agents 属性
    num_agents = len(env.action_space)
    print(f"Num Agents: {num_agents}")

    # [关键修复] 更新 all_args 中的 num_agents，确保 Policy/Mixer 使用正确的数量
    all_args.num_agents = num_agents

    # Policy Info 配置
    if all_args.share_policy:
        print("Sharing Policy...")
        policy_info = {
            'policy_0': {
                "cent_obs_dim": get_dim_from_space(env.share_observation_space[0]),
                "cent_act_dim": get_cent_act_dim(env.action_space),
                "obs_space": env.observation_space[0],
                "share_obs_space": env.share_observation_space[0],
                "act_space": env.action_space[0]
            }
        }

        def policy_mapping_fn(id):
            return 'policy_0'
    else:
        print("Initializing Independent Policies...")
        policy_info = {}
        for agent_id in range(num_agents):
            policy_info['policy_' + str(agent_id)] = {
                "cent_obs_dim": get_dim_from_space(env.share_observation_space[agent_id]),
                "cent_act_dim": get_dim_from_space(env.action_space[agent_id]),
                "obs_space": env.observation_space[agent_id],
                "share_obs_space": env.share_observation_space[agent_id],
                "act_space": env.action_space[agent_id]
            }

        def policy_mapping_fn(agent_id):
            return 'policy_' + str(agent_id)

    # 选择 Runner
    if all_args.algorithm_name in ["mqmix", "mvdn", "qmix", "vdn"]:
        # [MQMix/QMix] 使用原版 Runner (兼容性最好)
        print(f"Using Standard Runner for {all_args.algorithm_name}...")
        from offpolicy.runner.rnn.kill_chain_runner import KillChainRunner as Runner

    elif all_args.algorithm_name in ["rmaddpg", "rmatd3", "maddpg", "matd3"]:
        # [RMADDPG/RMATD3] 使用修复版 Runner (解决了动作维度 int/list 和切片问题)
        print(f"Using Fixed Runner (KillChainRunnerRMADDPG) for {all_args.algorithm_name}...")
        from offpolicy.runner.rnn.kill_chain_runner_rmaddpg import KillChainRunnerRMADDPG as Runner

    else:
        raise NotImplementedError(f"Algorithm {all_args.algorithm_name} is not supported.")

    config = {
        "args": all_args,
        "policy_info": policy_info,
        "policy_info": policy_info,
        "policy_mapping_fn": policy_mapping_fn,
        "env": env,
        "eval_env": eval_env,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
        "use_same_share_obs": all_args.use_same_share_obs,
        "use_available_actions": all_args.use_available_actions,
        "buffer_length": all_args.episode_length
    }

    runner = Runner(config=config)

    total_num_steps = 0
    while total_num_steps < all_args.num_env_steps:
        total_num_steps = runner.run()

    env.close()
    if eval_env is not None:
        eval_env.close()

    if all_args.use_wandb:
        run.finish()
    else:
        if hasattr(runner, 'writter'):
            runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
            runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])