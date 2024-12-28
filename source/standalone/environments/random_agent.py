# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to an environment with random action agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import yaml
from omegaconf import OmegaConf

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Isaac-Klask-v0", help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import time
import matplotlib.pyplot as plt
import numpy as np

from omni.isaac.lab.managers import SceneEntityCfg
import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.manager_based.klask import KlaskGoalEnvWrapper, CurriculumWrapper
from omni.isaac.lab_tasks.manager_based.klask.utils_manager_based import distance_player_ball


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    #env = KlaskGoalEnvWrapper(env)
    #cfg = OmegaConf.load('/home/idsc/IsaacLabKlask/tdmpc2/config_klask_finetune.yaml')
    #env = CurriculumWrapper(env, cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    rewards = []
    dists = []
    obs, info = env.reset()
    start_time = time.time()
    # simulate environment
    while simulation_app.is_running() and time.time() - start_time < 10.0:
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            #actions = (2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1)
            target_vel = obs["observation"][:, 8:10] - obs["observation"][:, [0, 1]]
            norm = torch.norm(target_vel, dim=-1)
            target_vel[:, 0] /= norm
            target_vel[:, 1] /= norm
            actions =  10 * np.random.rand() * target_vel
            actions = torch.concatenate((actions, actions), dim=-1)
            # actions = 0.5 * torch.ones(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            obs, rew, terminated, truncated, info = env.step(actions)
            rewards.append(rew.cpu().numpy())
            dist = torch.sqrt(torch.sum((obs["observation"][:, 8:10] - obs["observation"][:, [0, 1]]) ** 2, dim=-1))
            dists.append(dist.cpu().numpy())

    # close the simulator
    env.close()
    fig, ax = plt.subplots(2)
    ax[0].plot(dists)
    ax[1].plot(rewards)
    plt.show()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
