from tdmpc2 import TDMPC2
from common.parser import parse_cfg
import yaml
import numpy as np
import torch
from tensordict.tensordict import TensorDict

num_envs = 4
action_dim = 2
obs_dim = 12

def rand_act():
	return torch.rand(num_envs, action_dim)


class Test:
	def __init__(self, num_envs):
		self._tds = {_id: [] for _id in range(num_envs)}
		self.num_envs = num_envs

	def to_td(self, obs, action=None, reward=None, done=None):
			"""
				Creates a TensorDict for a new episode.
				
				:param obs: shape [num_envs, *obs_shape]
				:param action: shape [num_envs, action_dim]
				:param reward: shape [num_envs]
			"""
			if isinstance(obs, dict):
				num_envs = next(iter(obs.values())).shape[0]
				# Convert the dictionary into a list of tensordicts
				obs = [
					TensorDict(
						{key: value[i].unsqueeze(0) for key, value in obs.items()},
						batch_size=[]
					)
					for i in range(num_envs)
				]
			else:
				obs = obs.unsqueeze(1).cpu()
			if action is None:
				action = torch.full_like(rand_act(), float('nan'))
			if reward is None:
				reward = torch.full((action.shape[0],), float('nan'))
			action = action.unsqueeze(1).cpu()
			reward = reward.unsqueeze(1).cpu()
			
			for _id in range(self.num_envs):
				self._tds[_id].append(TensorDict(
					obs=obs[_id],
					action=action[_id],
					reward=reward[_id],
					batch_size=(1,)
				))
				if done[_id]:
					#self.buffer.add(torch.cat(self._tds[_id]))
					self._tds[_id] = []


obs = torch.rand(num_envs, obs_dim)
#obs = {
#	"obs": torch.rand(num_envs, obs_dim),
#	"goal": torch.rand(num_envs, obs_dim)
#}

action = torch.rand(num_envs, action_dim)
reward = torch.rand(num_envs)
done = torch.Tensor([False, False, False, True]).bool()

#test = Test(num_envs)
#test.to_td(obs, None, None, done)
#print(test._tds)