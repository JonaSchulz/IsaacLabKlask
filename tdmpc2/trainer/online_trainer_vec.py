from time import time

import numpy as np
import torch
from tensordict.tensordict import TensorDict
from trainer.base import Trainer


class OnlineTrainerVec(Trainer):
	"""Trainer class for single-task online TD-MPC2 training."""

	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self._step = 0
		self._ep_idx = 0
		self._start_time = time()

	def common_metrics(self):
		"""Return a dictionary of current metrics."""
		return dict(
			step=self._step,
			episode=self._ep_idx,
			total_time=time() - self._start_time,
		)

	def eval(self):
		"""Evaluate a TD-MPC2 agent."""
		ep_rewards, ep_successes = [], []
		for i in range(self.cfg.eval_episodes):
			obs, done, ep_reward, t = (self.env.reset(), 
							  torch.zeros(self.cfg.num_envs).bool(),
							  torch.zeros(self.cfg.num_envs).float(),
							  torch.zeros(self.cfg.num_envs).int()
							  )
			if self.cfg.save_video:
				self.logger.video.init(self.env, enabled=(i==0))
			while not done[0]:
				torch.compiler.cudagraph_mark_step_begin()
				action = self.agent.act(obs, t0=t==0, eval_mode=True)
				obs, reward, done, info = self.env.step(action)
				ep_reward += reward
				t += 1
				if self.cfg.save_video:
					self.logger.video.record(self.env)
			ep_rewards.append(ep_reward)
			ep_successes.append(info['success'])
			if self.cfg.save_video:
				self.logger.video.save(self._step)
		
		ep_rewards = torch.cat(ep_rewards).cpu().numpy()
		return dict(
			episode_reward=np.nanmean(ep_rewards),
			episode_success=np.nanmean(ep_successes),
		)		

	def to_td(self, obs, action=None, reward=None, done=None):
		"""
			Updates list of TensorDicts for each environment. For done environments, adds the episode
			to the buffer.
			
			:param obs: shape [num_envs, *obs_shape]
			:param action: shape [num_envs, action_dim]
			:param reward: shape [num_envs]
			:param done: shape [num_envs]
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
			action = torch.full_like(self.env.rand_act(), float('nan'))
		if reward is None:
			reward = torch.full((action.shape[0],), float('nan'))
		if done is None:
			done = torch.full((action.shape[0],), False)
		action = action.unsqueeze(1).cpu()
		reward = reward.unsqueeze(1).cpu()
		
		for _id in range(self.cfg.num_envs):
			self._tds[_id].append(TensorDict(
				obs=obs[_id],
				action=action[_id],
				reward=reward[_id],
				batch_size=(1,)
			))
			if done[_id]:
				self._ep_idx = self.buffer.add(torch.cat(self._tds[_id]))
				self._tds[_id] = []
		
	def train(self):
		"""Train a TD-MPC2 agent."""
		train_metrics, log, eval_next = {}, True, False
		self._tds = {_id: [] for _id in range(self.cfg.num_envs)}
		obs = self.env.reset()
		last_time = time()
		while self._step <= self.cfg.steps:
			# Evaluate agent periodically
			if self._step % (self.cfg.eval_freq * self.cfg.num_envs) == 0:
				eval_next = True
			if self._step % 1024 == 0:
				print(time() - last_time)
				last_time = time()

			# Reset environment
			# TODO
			if eval_next:
					eval_metrics = self.eval()
					eval_metrics.update(self.common_metrics())
					self.logger.log(eval_metrics, 'eval')
					eval_next = False

			# TODO: implement logging for vec env; currently temporary workaround by logging only first env
			if log:
				if self._step > 0:
					# update metrics with values from first env
					train_metrics.update(
						episode_reward=torch.tensor([td['reward'] for td in self._tds[0][1:]]).sum(),
						episode_success=info['success'],
					)
					train_metrics.update(self.common_metrics())
					self.logger.log(train_metrics, 'train')

				# obs = self.env.reset()
				# self.to_td(obs)

			# Collect experience
			if self._step > self.cfg.seed_steps:
				# TODO:
				action = self.agent.act(obs, t0=[len(self._tds[i])==1 for i in range(self.cfg.num_envs)])
			else:
				# TODO:
				action = self.env.rand_act()
			obs, reward, done, info = self.env.step(action)
			self.to_td(obs, action, reward, done)
			log = done[0]

			# Update agent
			if self._step >= self.cfg.seed_steps:
				if self._step == self.cfg.seed_steps:
					num_updates = self.cfg.seed_steps
					print('Pretraining agent on seed data...')
				else:
					num_updates = 1
				for _ in range(num_updates):
					_train_metrics = self.agent.update(self.buffer)
				train_metrics.update(_train_metrics)

			self._step += self.cfg.num_envs

		self.logger.finish(self.agent)
