import numpy as np
from collections import defaultdict
import torch
import gymnasium as gym
from gymnasium import Wrapper, ActionWrapper, spaces
from collections import OrderedDict
from stable_baselines3.common.vec_env.base_vec_env import VecEnv

from omni.isaac.lab_assets.klask import KLASK_PARAMS
from omni.isaac.lab_tasks.utils.wrappers.sb3 import Sb3VecEnvWrapper, process_sb3_cfg
from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv


class KlaskGoalEnvWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        
        self.player_in_goal_weight = KLASK_PARAMS["reward_player_in_goal"]
        self.goal_conceded_weight = KLASK_PARAMS["reward_goal_conceded"]
        self.goal_scored_weight = KLASK_PARAMS["reward_goal_scored"]
        self.ball_speed_weight = KLASK_PARAMS["reward_ball_speed"]
        self.proximity_to_ball_weight = KLASK_PARAMS["reward_proximity_to_ball"]
        self.dt = self.unwrapped.step_dt
        self.single_observation_space = self.unwrapped.single_observation_space
        self.single_action_space = spaces.Box(
            self.unwrapped.single_action_space.low[:2], 
            self.unwrapped.single_action_space.high[:2],
            shape=(2,), 
            dtype=self.unwrapped.single_action_space.dtype
        )
    
    def compute_reward(self, achieved_goal, desired_goal, info, observation, **kwargs):
        player_in_goal_reward = self.compute_player_in_goal_reward(observation)
        goal_conceded_reward = self.compute_goal_conceded_reward(observation)
        goal_reward = self.compute_goal_reward(achieved_goal, desired_goal)
        ball_speed_reward = self.compute_ball_speed_reward(observation)
        proximity_to_ball_reward = self.compute_proximity_to_ball_reward(observation)
        return [self.dt * (goal_reward + player_in_goal_reward + goal_conceded_reward + ball_speed_reward 
                           + proximity_to_ball_reward)]
    
    def compute_goal_reward(self, achieved_goal, desired_goal):
        # TODO: possibly need to unnormalize achieved and desired goal OR don't normalize achieved
        # and desired goal in the first place
        r = KLASK_PARAMS["opponent_goal"][2]
        v = np.sqrt(desired_goal[:, 2] ** 2 + desired_goal[:, 3] ** 2)
        ball_in_goal = (achieved_goal[:, 0] - desired_goal[:, 0]) ** 2 + (achieved_goal[:, 1] - desired_goal[:, 1]) ** 2 <= r ** 2
        ball_slow = ((achieved_goal[:, 2] ** 2 + achieved_goal[:, 3] ** 2 >= v ** 2) & 
                     (achieved_goal[:, 2] ** 2 + achieved_goal[:, 3] ** 2 <= (v + KLASK_PARAMS["max_ball_vel"]) ** 2))
        return self.goal_scored_weight * ball_in_goal * ball_slow
    
    def compute_player_in_goal_reward(self, observation):
        cx, cy, r = KLASK_PARAMS["player_goal"]
        player_in_goal = (observation[:, 0] - cx) ** 2 + (observation[:, 1] - cy) ** 2 <= r ** 2
        return self.player_in_goal_weight * player_in_goal

    def compute_goal_conceded_reward(self, observation):
        cx, cy, r = KLASK_PARAMS["player_goal"]
        ball_in_goal = (observation[:, 8] - cx) ** 2 + (observation[:, 9] - cy) ** 2 <= r ** 2
        ball_slow = observation[:, 10] ** 2 + observation[:, 11] ** 2 <= KLASK_PARAMS["max_ball_vel"] ** 2
        return self.goal_conceded_weight * ball_in_goal * ball_slow

    def compute_ball_speed_reward(self, observation):
        return self.ball_speed_weight * np.sqrt((observation[:, 10] ** 2 + observation[:, 11] ** 2))   

    def compute_proximity_to_ball_reward(self, observation):
        return  self.proximity_to_ball_weight * np.sqrt((observation[:, 0] - observation[:, 8]) ** 2 + (observation[:, 1] - observation[:, 9]) ** 2)


class KlaskSimpleEnvWrapper(Wrapper):

    def __init__(self, env):
        super().__init__(env)
        
        self.single_observation_space = self.unwrapped.single_observation_space["observation"]
        self.single_action_space = self.unwrapped.single_action_space
        #self.single_action_space = spaces.Box(
        #    self.unwrapped.single_action_space.low[:2], 
        #    self.unwrapped.single_action_space.high[:2],
        #    shape=(2,), 
        #    dtype=self.unwrapped.single_action_space.dtype
        #)
    
    def step(self, actions):
        obs, rew, terminated, truncated, info = self.env.step(actions)
        return obs["observation"], rew, terminated, truncated, info
    
    def reset(self):
        obs, info = self.env.reset()
        return obs["observation"], info


class KlaskSb3VecEnvWrapper(Sb3VecEnvWrapper):
    
    def __init__(self, env: KlaskGoalEnvWrapper | KlaskSimpleEnvWrapper):
        """Initialize the wrapper.

        Args:
            env: The environment to wrap around.

        Raises:
            ValueError: When the environment is not an instance of :class:`ManagerBasedRLEnv` or :class:`DirectRLEnv`.
        """
        # check that input is valid
        if not isinstance(env.unwrapped, ManagerBasedRLEnv) and not isinstance(env.unwrapped, DirectRLEnv):
            raise ValueError(
                "The environment must be inherited from ManagerBasedRLEnv or DirectRLEnv. Environment type:"
                f" {type(env)}"
            )
        # initialize the wrapper
        self.env = env
        # collect common information
        self.num_envs = self.unwrapped.num_envs
        self.sim_device = self.unwrapped.device
        self.render_mode = self.unwrapped.render_mode

        # obtain gym spaces
        # note: stable-baselines3 does not like when we have unbounded action space so
        #   we set it to some high value here. Maybe this is not general but something to think about.
        # observation_space = self.unwrapped.single_observation_space
        observation_space = self.env.single_observation_space
        action_space = self.env.single_action_space
        if isinstance(action_space, gym.spaces.Box) and not action_space.is_bounded("both"):
            action_space = gym.spaces.Box(low=-100, high=100, shape=action_space.shape)

        # initialize vec-env
        VecEnv.__init__(self, self.num_envs, observation_space, action_space)
        # add buffer for logging episodic information
        self._ep_rew_buf = torch.zeros(self.num_envs, device=self.sim_device)
        self._ep_len_buf = torch.zeros(self.num_envs, device=self.sim_device)

    def _process_obs(self, obs_dict: torch.Tensor | dict[str, torch.Tensor]) -> np.ndarray | dict[str, np.ndarray]:
        """Convert observations into NumPy data type."""
        obs = obs_dict

        # note: ManagerBasedRLEnv uses torch backend (by default).
        if isinstance(obs, dict):
            for key, value in obs.items():
                obs[key] = value.detach().cpu().numpy()
        elif isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()
        else:
            raise NotImplementedError(f"Unsupported data type: {type(obs)}")
        return obs
    

class KlaskSingleEnvWrapper(Wrapper):

    def __init__(self, env: ManagerBasedRLEnv, single_player=True):
        super().__init__(env)
        self.single_player = single_player
        assert env.num_envs == 1
        if single_player:
            self._action_space = spaces.Box(
                self.unwrapped.single_action_space.low[:2], 
                self.unwrapped.single_action_space.high[:2],
                shape=(2,), 
                dtype=self.unwrapped.single_action_space.dtype
            )
        else:
            self._action_space = self.unwrapped.single_action_space

    def step(self, actions):
        actions = torch.from_numpy(actions).unsqueeze(0)
        if self.single_player:
            opponent_actions = 20.0 * (2 * torch.rand(actions.shape, device=actions.device) - 1)
            actions = torch.cat([actions, opponent_actions], dim=-1)
        obs, rew, terminated, truncated, info = self.env.step(actions)
        terminated = terminated[0] | truncated[0]
        if type(obs) is dict:
            if 'observation' in obs.keys():
                obs = obs['observation']
            else:
                obs = obs['policy']
        return obs[0].cpu().numpy(), rew[0].cpu().numpy(), terminated.cpu().numpy(), info
    
    def reset(self):
        obs, _ = self.env.reset()
        if type(obs) is dict:
            if 'observation' in obs.keys():
                obs = obs['observation']
            else:
                obs = obs['policy']
        return obs[0].cpu().numpy()
    

class KlaskTDMPCWrapper(Wrapper):

    def __init__(self, env: ManagerBasedRLEnv, single_player=True):
        super().__init__(env)
        self.single_player = single_player
    
    def step(self, actions):
        if self.single_player:
            opponent_actions = 2 * torch.rand(actions.shape, device=actions.device) - 1
            actions = torch.cat([actions, opponent_actions], dim=-1)
        obs, rew, terminated, truncated, info = self.env.step(actions)
        terminated = terminated | truncated
        if type(obs) is dict:
            if 'observation' in obs.keys():
                obs = obs['observation']
            else:
                obs = obs['policy']
        return obs.cpu(), rew.cpu(), terminated, info
    
    def reset(self):
        obs, _ = self.env.reset()
        if type(obs) is dict:
            if 'observation' in obs.keys():
                return obs['observation'].cpu()
            else:
                return obs['policy'].cpu()
        return obs.cpu()
        