"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with Stable-Baselines3.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from omni.isaac.lab_tasks.manager_based.klask import (
	KlaskTDMPCWrapper,
    KlaskEnvCfg
)
from omni.isaac.lab_tasks.manager_based.classic.cartpole import CartpoleEnvCfg

import os
os.environ['MUJOCO_GL'] = 'egl'
os.environ['LAZY_LEGACY_OP'] = '0'
os.environ['TORCHDYNAMO_INLINE_INBUILT_NN_MODULES'] = "1"
os.environ['TORCH_LOGS'] = "+recompiles"
import warnings
warnings.filterwarnings('ignore')
import torch

import hydra
from termcolor import colored

from common.parser import parse_cfg
from common.seed import set_seed
from common.buffer import Buffer
from envs import make_env
from tdmpc2_vec import TDMPC2Vec
from trainer.offline_trainer import OfflineTrainer
from trainer.online_trainer_vec import OnlineTrainerVec
from common.logger import Logger
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')


@hydra.main(config_name='config_klask', config_path='.')
def train(cfg: dict):
	"""
	Script for training single-task / multi-task TD-MPC2 agents.

	Most relevant args:
		`task`: task name (or mt30/mt80 for multi-task training)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]` (default: 5)
		`steps`: number of training/environment steps (default: 10M)
		`seed`: random seed (default: 1)

	See config.yaml for a full list of args.

	Example usage:
	```
		$ python train.py task=mt80 model_size=48
		$ python train.py task=mt30 model_size=317
		$ python train.py task=dog-run steps=7000000
	```
	"""
	assert torch.cuda.is_available()
	assert cfg.steps > 0, 'Must train for at least 1 step.'
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	print(colored('Work dir:', 'yellow', attrs=['bold']), cfg.work_dir)

	# make env:
	env_cfg = KlaskEnvCfg()
	#env_cfg = CartpoleEnvCfg()
	env_cfg.scene.num_envs = cfg.num_envs if cfg.num_envs is not None else env_cfg.scene.num_envs
	env = gym.make(cfg.task, cfg=env_cfg, render_mode='rgb_array' if cfg.save_video else None)
	env = KlaskTDMPCWrapper(env, single_player=cfg.single_player)
	cfg.obs_shape = {'state': (12,)}
	cfg.episode_length = env.unwrapped.max_episode_length
	cfg.seed_steps = max(1000, 5*cfg.episode_length)

	trainer_cls = OfflineTrainer if cfg.multitask else OnlineTrainerVec
	trainer = trainer_cls(
		cfg=cfg,
		env=env,
		agent=TDMPC2Vec(cfg),
		buffer=Buffer(cfg),
		logger=Logger(cfg),
	)
	trainer.train()
	print('\nTraining completed successfully')


if __name__ == '__main__':
	train()
