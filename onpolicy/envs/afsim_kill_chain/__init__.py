"""
AFSIM Kill Chain Environment Package
基于原 kill_chain 环境, 用 AFSIM 替换物理仿真层。
"""
from .config import Config, AFSIMConfig, CURRENT_SCENARIO
from .core import (
    EndogenousNode,
    TaskEntity,
    Interaction,
    Message,
    NodeType,
    TaskStage
)
from .node_mapping import NodeMapper, PlatformInfo
from .afsim_bridge import AFSIMBridge, NodeSnapshot
from .AFSIMKillChainEnv import AFSIMKillChainEnv
from .env_wrappers import ShareDummyVecEnv, make_train_env, make_eval_env

__all__ = [
    'Config', 'AFSIMConfig', 'CURRENT_SCENARIO',
    'EndogenousNode', 'TaskEntity', 'Interaction', 'Message',
    'NodeType', 'TaskStage',
    'NodeMapper', 'PlatformInfo',
    'AFSIMBridge', 'NodeSnapshot',
    'AFSIMKillChainEnv',
    'ShareDummyVecEnv', 'make_train_env', 'make_eval_env',
]

__version__ = '3.0.0-afsim'