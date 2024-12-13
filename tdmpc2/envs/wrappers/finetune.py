import gym
import numpy as np
import torch


class FinetuneWrapper(gym.Wrapper):
	"""
	Wrapper for multi-task environments.
	"""

	def __init__(self, cfg, env):
		super().__init__(env)
		self.cfg = cfg
		self._obs_shape = cfg.obs_shape[cfg.obs]

	def _pad_obs(self, obs):
		if obs.shape != self._obs_shape:
			obs = torch.cat((obs, torch.zeros(self._obs_shape[0]-obs.shape[0], dtype=obs.dtype, device=obs.device)))
		return obs
	
	def reset(self):
		return self._pad_obs(self.env.reset())

	def step(self, action):
		obs, reward, done, info = self.env.step(action[:self.env.action_space.shape[0]])
		return self._pad_obs(obs), reward, done, info
