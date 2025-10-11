#!/usr/bin/env python3

"""
Trains a 2-layer MLP policy with MAML-TRPO on the Meta-World ML10 benchmark.

This script is an adaptation of the original learn2learn MAML-TRPO example.
"""

import random
from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import torch
import metaworld
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import learn2learn as l2l
from examples.rl.policies import DiagNormalPolicy # Assuming policies.py is in examples/rl/

class MetaWorldML10(l2l.gym.MetaEnv):
    """
    Wrapper for the Meta-World ML10 benchmark to make it compatible
    with the learn2learn MetaEnv interface.
    """
    def __init__(self, seed=None):
        self.ml10 = metaworld.ML10(seed=seed)
        self.train_tasks = self.ml10.train_tasks
        self._active_env = None
        # Set an initial task to define spaces
        self.set_task(self.sample_tasks(1)[0])

    @property
    def observation_space(self):
        return self._active_env.observation_space

    @property
    def action_space(self):
        return self._active_env.action_space

    def set_task(self, task):
        """Sets the active environment based on the task description."""
        env_name = task.env_name
        env_cls = self.ml10.train_classes[env_name]
        self._active_env = env_cls()
        self._active_env.set_task(task)
        self._active_env.reset() 
        return True

    def sample_tasks(self, num_tasks):
        """Samples a list of tasks from the ML10 training set."""
        return random.choices(self.train_tasks, k=num_tasks)

    def step(self, action):
        obs, reward, terminated, truncated, info = self._active_env.step(action)
        done = terminated or truncated
        return obs, reward, done, info
    
    def reset(self, **kwargs):
        obs, info = self._active_env.reset(**kwargs)
        return obs

    def render(self, **kwargs):
        return self._active_env.render(**kwargs)


def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    """Computes GAE."""
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau=tau,
                                       gamma=gamma,
                                       rewards=rewards,
                                       dones=dones,
                                       values=bootstraps,
                                       next_value=next_value)


def maml_a2c_loss(train_episodes, learner, baseline, gamma, tau):
    """Computes the A2C loss for a given task."""
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)


def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    """Performs a single MAML adaptation step."""
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone, adapt_lr, gradients)


def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr):
    """Computes the TRPO surrogate loss across a batch of tasks."""
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in tqdm(zip(iteration_replays, iteration_policies),
                                         total=len(iteration_replays),
                                         desc='Surrogate Loss',
                                         leave=False):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        # Fast Adapt
        for train_episodes in train_replays:
            new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,
                                        baseline, gamma, tau, first_order=False)

        # Useful values for validation
        states = valid_episodes.state()
        actions = valid_episodes.action()
        rewards = valid_episodes.reward()
        dones = valid_episodes.done()
        next_states = valid_episodes.next_state()
        
        # Compute KL divergence
        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        # Compute Surrogate Loss
        advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    
    mean_kl /= len(iteration_replays)
    mean_loss /= len(iteration_replays)
    return mean_loss, mean_kl


def main(
        adapt_lr=0.1,
        meta_lr=1.0,
        adapt_steps=1,
        num_iterations=500,
        meta_bsz=10,  # Number of tasks per meta-batch (ML10 has 10 train tasks)
        adapt_bsz=20, # Number of episodes per adaptation step
        tau=1.00,
        gamma=0.99,
        seed=42,
        num_workers=4,
        cuda=True,
):
    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = 'cuda:2' if cuda and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if cuda:
        torch.cuda.manual_seed(seed)

    def make_env():
        env = MetaWorldML10(seed=seed)
        env = ch.envs.ActionSpaceScaler(env)
        return env

    dummy_env = make_env()
    state_size = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.shape[0]

    env = l2l.gym.AsyncVectorEnv([make_env for _ in range(num_workers)])
    env.seed(seed)
    env = ch.envs.Torch(env)
    
    policy = DiagNormalPolicy(state_size, action_size, device=device)
    if cuda:
        policy.to(device)
    baseline = LinearValue(state_size, action_size)

    for iteration in range(num_iterations):
        iteration_reward = 0.0
        iteration_replays = []
        iteration_policies = []

        # Sample tasks and collect data
        tasks = env.sample_tasks(meta_bsz)
        for task_config in tqdm(tasks, leave=False, desc='Data Collection'):
            clone = deepcopy(policy)
            env.set_task(task_config)
            env.reset()
            task_runner = ch.envs.Runner(env)
            task_replay = []

            # Adaptation phase
            for step in range(adapt_steps):
                train_episodes = task_runner.run(clone, episodes=adapt_bsz)
                if cuda:
                    train_episodes = train_episodes.to(device, non_blocking=True)
                clone = fast_adapt_a2c(clone, train_episodes, adapt_lr,
                                       baseline, gamma, tau, first_order=True)
                task_replay.append(train_episodes)

            # Validation phase
            valid_episodes = task_runner.run(clone, episodes=adapt_bsz)
            task_replay.append(valid_episodes)
            iteration_reward += valid_episodes.reward().sum().item() / adapt_bsz
            iteration_replays.append(task_replay)
            iteration_policies.append(clone)

        # Print statistics
        print(f'\n--- Iteration {iteration} ---')
        adaptation_reward = iteration_reward / meta_bsz
        print(f'Average Adaptation Reward: {adaptation_reward:.4f}')

        # TRPO meta-optimization
        backtrack_factor = 0.8
        ls_max_steps = 15
        max_kl = 0.01
        if cuda:
            policy.to(device, non_blocking=True)
            baseline.to(device, non_blocking=True)
            iteration_replays = [
                [replay.to(device, non_blocking=True) for replay in task_replays]
                for task_replays in iteration_replays
            ]

        # Compute meta-gradients with Conjugate Gradient
        old_loss, old_kl = meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr)
        grad = autograd.grad(old_loss, policy.parameters(), retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])
        
        Fvp = trpo.hessian_vector_product(old_kl, policy.parameters())
        step_direction = trpo.conjugate_gradient(Fvp, grad)
        
        shs = 0.5 * torch.dot(step_direction, Fvp(step_direction))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = step_direction / lagrange_multiplier
        
        step_ = [torch.zeros_like(p.data) for p in policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_
        del old_kl, Fvp, grad
        old_loss.detach_()

        # Line search to update policy
        for ls_step in range(ls_max_steps):
            stepsize = backtrack_factor ** ls_step * meta_lr
            clone = deepcopy(policy)
            for p, u in zip(clone.parameters(), step):
                p.data.add_(u.data, alpha=-stepsize)
                
            new_loss, kl = meta_surrogate_loss(iteration_replays, iteration_policies, clone, baseline, tau, gamma, adapt_lr)
            if new_loss < old_loss and kl < max_kl:
                for p, u in zip(policy.parameters(), step):
                    p.data.add_(u.data, alpha=-stepsize)
                break

if __name__ == '__main__':
    main()