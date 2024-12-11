import gymnasium as gym

from . import agents
from .klask_env_cfg import KlaskEnvCfg
from .klask_env_wrapper import *
from .klask_her import SubtaskHerReplayBuffer
from .klask_algorithms import TwoPlayerSAC, TwoPlayerPPO

##
# Register Gym environment.
##

gym.register(
    id="Isaac-Klask-v0",
    entry_point="omni.isaac.lab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": KlaskEnvCfg,
        "sb3_cfg_entry_point": f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)
