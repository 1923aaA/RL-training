"""
Kill Chain Environment Package (Updated Version)
杀伤链多智能体强化学习环境（4维能力+通信时延版本）
"""

from .KillChain_Env import KillChainEnv
from .config import Config
from .core import (
    EndogenousNode,
    TaskEntity,
    Interaction,
    Message,  # 新增
    NodeType,
    TaskStage
)
from .env_wrappers import (
    DummyVecEnv,
    ShareSubprocVecEnv,
    make_train_env,
    make_eval_env
)

__all__ = [
    'KillChainEnv',
    'Config',
    'EndogenousNode',
    'TaskEntity',
    'Interaction',
    'Message',  # 新增
    'NodeType',
    'TaskStage',
    'DummyVecEnv',
    'ShareSubprocVecEnv',
    'make_train_env',
    'make_eval_env'
]

__version__ = '2.0.0'  # 更新版本号