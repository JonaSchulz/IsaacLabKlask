import torch
import torch.nn.functional as F

from common import math
from common.scale import RunningScale
from common.world_model import WorldModel
from tensordict import TensorDict
from tdmpc2 import TDMPC2

class TDMPC2Vec(TDMPC2):
	
	def __init__(self, cfg):
		torch.nn.Module.__init__(self)
		self.cfg = cfg
		self.device = torch.device('cuda:0')
		self.model = WorldModel(cfg).to(self.device)
		self.optim = torch.optim.Adam([
			{'params': self.model._encoder.parameters(), 'lr': self.cfg.lr*self.cfg.enc_lr_scale},
			{'params': self.model._dynamics.parameters()},
			{'params': self.model._reward.parameters()},
			{'params': self.model._Qs.parameters()},
			{'params': self.model._task_emb.parameters() if self.cfg.multitask else []
			 }
		], lr=self.cfg.lr, capturable=True)
		self.pi_optim = torch.optim.Adam(self.model._pi.parameters(), lr=self.cfg.lr, eps=1e-5, capturable=True)
		self.model.eval()
		self.scale = RunningScale(cfg)
		self.cfg.iterations += 2*int(cfg.action_dim >= 20) # Heuristic for large action spaces
		self.discount = torch.tensor(
			[self._get_discount(ep_len) for ep_len in cfg.episode_lengths], device='cuda:0'
		) if self.cfg.multitask else self._get_discount(cfg.episode_length)
		self._prev_mean = {
			i: torch.nn.Buffer(torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)) for i in range(cfg.num_envs)
		}		
		if cfg.compile:
			print('Compiling update function with torch.compile...')
			self._update = torch.compile(self._update, mode="reduce-overhead")
	
	@torch.no_grad()
	def act(self, obs, t0=False, eval_mode=False, task=None):
		"""
		Select an action by planning in the latent space of the world model.

		Args:
			obs (torch.Tensor): Observation from the environment, shape [num_envs, *obs_dim]
			t0 (List[int]): Whether this is the first observation in the episode, shape [num_envs]
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (int): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		obs = obs.to(self.device, non_blocking=True)
		if type(t0) is bool:
			t0 = [t0 for _ in range(self.cfg.num_envs)]
		if task is not None:
			task = torch.tensor([task], device=self.device)
		if self.cfg.mpc:
			a = torch.empty(self.cfg.num_envs, self.cfg.action_dim, device=self.device, requires_grad=False)
			for i in range(self.cfg.num_envs):
				a[i] = self.plan(obs[i].unsqueeze(0), i, t0=t0[i], eval_mode=eval_mode, task=task)
		else:
			z = self.model.encode(obs, task)
			a = self.model.pi(z, task)[int(not eval_mode)]
		return a.cpu()
	
	@torch.no_grad()
	def _plan(self, obs, env_id, t0=False, eval_mode=False, task=None):
		"""
		Plan a sequence of actions using the learned world model.

		Args:
			z (torch.Tensor): Latent state from which to plan.
			env_id (int)
			t0 (bool): Whether this is the first observation in the episode.
			eval_mode (bool): Whether to use the mean of the action distribution.
			task (Torch.Tensor): Task index (only used for multi-task experiments).

		Returns:
			torch.Tensor: Action to take in the environment.
		"""
		# Sample policy trajectories
		z = self.model.encode(obs, task)
		if self.cfg.num_pi_trajs > 0:
			pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs, self.cfg.action_dim, device=self.device)
			_z = z.repeat(self.cfg.num_pi_trajs, 1)
			for t in range(self.cfg.horizon-1):
				pi_actions[t] = self.model.pi(_z, task)[1]
				_z = self.model.next(_z, pi_actions[t], task)
			pi_actions[-1] = self.model.pi(_z, task)[1]

		# Initialize state and parameters
		z = z.repeat(self.cfg.num_samples, 1)
		mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
		std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
		if not t0:
			mean[:-1] = self._prev_mean[env_id][1:]
		actions = torch.empty(self.cfg.horizon, self.cfg.num_samples, self.cfg.action_dim, device=self.device)
		if self.cfg.num_pi_trajs > 0:
			actions[:, :self.cfg.num_pi_trajs] = pi_actions

		# Iterate MPPI
		for _ in range(self.cfg.iterations):

			# Sample actions
			r = torch.randn(self.cfg.horizon, self.cfg.num_samples-self.cfg.num_pi_trajs, self.cfg.action_dim, device=std.device)
			actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
			actions_sample = actions_sample.clamp(-1, 1)
			actions[:, self.cfg.num_pi_trajs:] = actions_sample
			if self.cfg.multitask:
				actions = actions * self.model._action_masks[task]

			# Compute elite actions
			value = self._estimate_value(z, actions, task).nan_to_num(0)
			elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
			elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

			# Update parameters
			max_value = elite_value.max(0).values
			score = torch.exp(self.cfg.temperature*(elite_value - max_value))
			score = score / score.sum(0)
			mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
			std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
			std = std.clamp(self.cfg.min_std, self.cfg.max_std)
			if self.cfg.multitask:
				mean = mean * self.model._action_masks[task]
				std = std * self.model._action_masks[task]

		# Select action
		rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
		actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
		a, std = actions[0], std[0]
		if not eval_mode:
			a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
		self._prev_mean[env_id].copy_(mean)
		return a.clamp(-1, 1)
	
	# @torch.no_grad()
	# def _plan(self, obs, t0=False, eval_mode=False, task=None):
	# 	"""
	# 	Plan a sequence of actions using the learned world model.

	# 	Args:
	# 		obs (torch.Tensor): Observation from which to plan, shape [num_envs, *obs_shape]
	# 		t0 (torch.Tensor): Whether this is the first observation in the episode, shape [num_envs]
	# 		eval_mode (bool): Whether to use the mean of the action distribution.
	# 		task (Torch.Tensor): Task index (only used for multi-task experiments).

	# 	Returns:
	# 		torch.Tensor: Action to take in the environment.
	# 	"""
	# 	# Sample policy trajectories
	# 	z = self.model.encode(obs, task)
	# 	num_envs = obs.shape[0]
	# 	if self.cfg.num_pi_trajs > 0:
	# 		pi_actions = torch.empty(self.cfg.horizon, self.cfg.num_pi_trajs * num_envs, self.cfg.action_dim, device=self.device)
	# 		_z = z.repeat(self.cfg.num_pi_trajs, 1)		# shape [num_pi_trajs * num_envs, latent_dim]
	# 		for t in range(self.cfg.horizon-1):
	# 			pi_actions[t] = self.model.pi(_z, task)[1]
	# 			_z = self.model.next(_z, pi_actions[t], task)
	# 		pi_actions[-1] = self.model.pi(_z, task)[1]

	# 	# Initialize state and parameters
	# 	z = z.repeat(self.cfg.num_samples, 1)	# shape [num_samples * num_envs, latent_dim]
	# 	mean = torch.zeros(self.cfg.horizon, self.cfg.action_dim, device=self.device)
	# 	std = torch.full((self.cfg.horizon, self.cfg.action_dim), self.cfg.max_std, dtype=torch.float, device=self.device)
	# 	if not t0:
	# 		mean[:-1] = self._prev_mean[1:]
	# 	actions = torch.empty(self.cfg.horizon, self.cfg.num_samples * num_envs, self.cfg.action_dim, device=self.device)
	# 	if self.cfg.num_pi_trajs > 0:
	# 		actions[:, :self.cfg.num_pi_trajs * num_envs] = pi_actions

	# 	# Iterate MPPI
	# 	for _ in range(self.cfg.iterations):

	# 		# Sample actions
	# 		r = torch.randn(self.cfg.horizon, (self.cfg.num_samples-self.cfg.num_pi_trajs) * num_envs, self.cfg.action_dim, device=std.device)
	# 		actions_sample = mean.unsqueeze(1) + std.unsqueeze(1) * r
	# 		actions_sample = actions_sample.clamp(-1, 1)
	# 		actions[:, self.cfg.num_pi_trajs:] = actions_sample
	# 		if self.cfg.multitask:
	# 			actions = actions * self.model._action_masks[task]

	# 		# Compute elite actions
	# 		value = self._estimate_value(z, actions, task).nan_to_num(0)
	# 		elite_idxs = torch.topk(value.squeeze(1), self.cfg.num_elites, dim=0).indices
	# 		elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

	# 		# Update parameters
	# 		max_value = elite_value.max(0).values
	# 		score = torch.exp(self.cfg.temperature*(elite_value - max_value))
	# 		score = score / score.sum(0)
	# 		mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
	# 		std = ((score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1) / (score.sum(0) + 1e-9)).sqrt()
	# 		std = std.clamp(self.cfg.min_std, self.cfg.max_std)
	# 		if self.cfg.multitask:
	# 			mean = mean * self.model._action_masks[task]
	# 			std = std * self.model._action_masks[task]

	# 	# Select action
	# 	rand_idx = math.gumbel_softmax_sample(score.squeeze(1))  # gumbel_softmax_sample is compatible with cuda graphs
	# 	actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
	# 	a, std = actions[0], std[0]
	# 	if not eval_mode:
	# 		a = a + std * torch.randn(self.cfg.action_dim, device=std.device)
	# 	self._prev_mean.copy_(mean)
	# 	return a.clamp(-1, 1)
	