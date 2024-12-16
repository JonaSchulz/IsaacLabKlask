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

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab_tasks.manager_based.klask import KlaskGoalEnvWrapper, CurriculumWrapper


def main():
    """Random actions agent with Isaac Lab environment."""
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    #env = KlaskGoalEnvWrapper(env)
    cfg = OmegaConf.load('/home/idsc/IsaacLabKlask/tdmpc2/config_klask_finetune.yaml')
    env = CurriculumWrapper(env, cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions from -1 to 1
            actions = (2 * torch.rand(env.action_space.shape, device=env.unwrapped.device) - 1)
            # actions = 0.5 * torch.ones(env.action_space.shape, device=env.unwrapped.device)
            # apply actions
            obs, rew, terminated, truncated, info = env.step(actions)
            if terminated[0] or truncated[0]:
                klask = env.env.scene.articulations["klask"]
                print(f"Joint names: {klask.data.joint_names}")
                print(f"Peg 1 Joint Position: {klask.data.joint_pos}")
                print(f"Body names: {klask.data.body_names}")
                print(f"Peg 1 Body Position: {klask.data.body_pos_w[0]}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
