#!/usr/bin/env python3

"""
Trains a 2-layer MLP policy with MAML-TRPO on the Meta-World ML10 benchmark.

This script is an adaptation of the original learn2learn MAML-TRPO example.
"""

import random
import os
from copy import deepcopy

import cherry as ch
import gym
import numpy as np
import torch
import metaworld
import wandb 
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from torch.utils.data import SubsetRandomSampler, BatchSampler
from tqdm import tqdm
from typing import Iterator, Dict, Any, List
from collections import defaultdict

import learn2learn as l2l
from examples.rl.policies import DiagNormalPolicy # Assuming policies.py is in examples/rl/

class _InfiniteSampler:
    """A helper class that creates an infinite iterator to yield random batches."""
    def __init__(self, items: List[Any], batch_size: int):
        self.items = items
        self.num_items = len(items)
        self.batch_size = batch_size

        # Guarantees each item appears at least floor(bsz / num_items) times
        self.base_batch = self.items * (self.batch_size // self.num_items)
        self.effective_batch_size = self.batch_size - len(self.base_batch)

        if self.effective_batch_size > 0:
            sampler = SubsetRandomSampler(range(self.num_items))
            self.batch_sampler = BatchSampler(sampler, batch_size=self.effective_batch_size, drop_last=True)
            self.iterator = iter(self.batch_sampler)
        else:
            self.iterator = None # Not needed if batch is just repetitions

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> List[Any]:
        if self.effective_batch_size == 0:
            return self.base_batch.copy() # Return copies to avoid mutation issues

        try:
            indices = next(self.iterator)
        except StopIteration:
            # Refill the iterator when it's exhausted
            self.iterator = iter(self.batch_sampler)
            indices = next(self.iterator)
        
        sampled_items = [self.items[i] for i in indices]
        return sampled_items + self.base_batch

class BalancedTaskSampler:
    """
    Samples a batch of tasks from a metaworld benchmark, ensuring that the
    number of tasks from each environment type is as balanced as possible.
    """
    def __init__(self, benchmark, batch_size: int, test: bool =False):
        self.batch_size = batch_size
        
        # 1. Group tasks by their environment name
        classes = benchmark.test_classes if test else benchmark.train_classes
        tasks = benchmark.test_tasks if test else benchmark.train_tasks
        env_names = list(classes.keys())
        tasks_by_env = defaultdict(list)
        for task in tasks:
            tasks_by_env[task.env_name].append(task)
            
        # 2. Create a sampler for environment types (e.g., 'reach-v2', 'push-v2')
        self.env_type_sampler = _InfiniteSampler(env_names, self.batch_size)
        
        # 3. Create a sampler for each environment's specific tasks (e.g., different goals)
        self.task_samplers = {
            name: _InfiniteSampler(tasks, 1) for name, tasks in tasks_by_env.items()
        }

    def __iter__(self) -> Iterator:
        return self
    
    def __next__(self) -> List[Any]:
        """Samples one balanced batch of tasks."""
        # Stage 1: Sample a balanced batch of environment names
        sampled_env_types = next(self.env_type_sampler)
        random.shuffle(sampled_env_types) # Shuffle to mix base and sampled items
        
        # Stage 2: For each env name, sample one specific task
        # The internal sampler yields a list of size 1, so we take the first element.
        batch = [next(self.task_samplers[name])[0] for name in sampled_env_types]
        return batch

class MetaWorldML10(l2l.gym.MetaEnv):
    """
    Wrapper for the Meta-World ML10 benchmark to make it compatible
    with the learn2learn MetaEnv interface.
    """
    def __init__(self, seed=None, test=False):
        self.test = test
        self.ml10 = metaworld.ML10(seed=seed)
        self.train_tasks = self.ml10.train_tasks
        self._active_env = None
        # Set an initial task to define spaces
        task = self.ml10.test_tasks if test else self.ml10.train_tasks
        self.set_task(task[0])
       
    @property
    def observation_space(self):
        return self._active_env.observation_space

    @property
    def action_space(self):
        return self._active_env.action_space

    def set_task(self, task):
        """Sets the active environment based on the task description."""
        env_name = task.env_name
        classes = self.ml10.test_classes if self.test else self.ml10.train_classes
        env_cls = classes[env_name]
        self._active_env = env_cls()
        self._active_env.set_task(task)
        self._active_env.reset() 
        return True

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

def make_env(test=False):
    def fn():
        env = MetaWorldML10(seed=42, test=test)
        env = ch.envs.ActionSpaceScaler(env)
        return env
    return fn

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
        num_workers=10,
        cuda=True,
):
    cuda = bool(cuda)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_name = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    if cuda:
        torch.cuda.manual_seed(seed)

    # Initialize W&B
    wandb.init(
        project="l2l-metaworld-ml10-maml-trpo",
        config={
            "adapt_lr": adapt_lr,
            "meta_lr": meta_lr,
            "adapt_steps": adapt_steps,
            "num_iterations": num_iterations,
            "meta_bsz": meta_bsz,
            "adapt_bsz": adapt_bsz,
            "tau": tau,
            "gamma": gamma,
            "seed": seed,
        }
    )

    benchmark = metaworld.ML10(seed=seed)
    task_sampler = BalancedTaskSampler(benchmark, batch_size=meta_bsz)

    dummy_task = next(task_sampler)[0]
    dummy_env = benchmark.train_classes[dummy_task.env_name]()
    dummy_env.set_task(dummy_task)
    state_size = dummy_env.observation_space.shape[0]
    action_size = dummy_env.action_space.shape[0]

    env = l2l.gym.AsyncVectorEnv([make_env() for _ in range(num_workers)])
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
        tasks = next(task_sampler)
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

        if iteration % 30 == 0:
            evaluate(benchmark, policy, baseline, adapt_lr, gamma, tau, num_workers, seed, cuda)

    path = "model/maml_ml.pth"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        'policy_state_dict': policy.state_dict(),
        'baseline_state_dict': baseline.state_dict(),
    }
    
    torch.save(checkpoint, path)
    print(f"Checkpoint saved successfully to {path}")

def evaluate(benchmark, policy, baseline, adapt_lr, gamma, tau, n_workers, seed, cuda):
    device_name = 'cpu'
    if cuda:
        device_name = 'cuda'
    device = torch.device(device_name)

    # Parameters
    adapt_steps = 3
    adapt_bsz = 10
    n_eval_tasks = 10

    tasks_reward = 0

    env = make_env(test=True)()
    env = ch.envs.Torch(env)
    benchmark = metaworld.ML10(seed=seed)
    task_sampler = BalancedTaskSampler(benchmark, batch_size=n_eval_tasks, test=True)
    results_by_class = defaultdict(list)
    for i, task in enumerate(next(task_sampler)):
        clone = deepcopy(policy)
        clone.to(device)
        env.set_task(task)
        env.reset()
        task_run = ch.envs.Runner(env)

        # Adapt
        for step in range(adapt_steps):
            adapt_episodes = task_run.run(clone, episodes=adapt_bsz)
            if cuda:
                adapt_episodes = adapt_episodes.to(device, non_blocking=True)
            clone = fast_adapt_a2c(clone, adapt_episodes, adapt_lr, baseline, gamma, tau, first_order=True)

        eval_episodes = task_run.run(clone, episodes=adapt_bsz)

        task_reward = eval_episodes.reward().sum().item() / adapt_bsz
        '''
        Change the run method in the Runner class
        state, reward, done, step_info = self.env.step(action)
        if isinstance(step_info, tuple):
            step_info = step_info[0]
        info["success"] = step_info["success"]
        '''
        task_success = eval_episodes.success().cpu().numpy().flatten()
        task_done = eval_episodes.done().cpu().numpy().flatten()
        num_episodes, successful_episodes, start_idx = 0, 0, 0
        end_indices = np.where(task_done == 1)[0]
        if len(end_indices) > 0:
            num_episodes = len(end_indices)
            for end_idx in end_indices:
                if np.any(task_success[start_idx:end_idx + 1] > 0):
                    successful_episodes += 1
                start_idx = end_idx + 1
            task_success_rate = successful_episodes / num_episodes
        else:
            task_success_rate = 0.0
        print("Success", task_success_rate)
        results_by_class[f"{task.env_name}_success_rate"].append(task_success_rate)
        tasks_reward += task_reward

    final_eval_reward = tasks_reward / n_eval_tasks
    
    print(f"Average reward over {n_eval_tasks} test tasks: {final_eval_reward}")
    num_train_tasks = 5
    for i in range(adapt_bsz//num_train_tasks):
        res = {}
        for env_name, successes in results_by_class.items():
            res[env_name] = successes[i]
        wandb.log(res)
    return final_eval_reward


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    main(
        num_iterations=600,
        adapt_lr=0.1119,
        meta_lr=0.9987,
        adapt_steps=2,
        meta_bsz=10,
        adapt_bsz=40,
        tau=0.9941,
        gamma=0.9886,
        seed=42,
        num_workers=10,
        cuda=True,
    )


