#!/usr/bin/env python3

"""
Core implementation of the MAML-TRPO algorithm, encapsulated in a reusable class.
This script can be run directly for a single experiment.
"""

import random
from copy import deepcopy
import os

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm

import learn2learn as l2l
from examples.rl.policies import DiagNormalPolicy

# --- Helper Functions (unchanged from original script) ---

def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
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
    states = train_episodes.state()
    actions = train_episodes.action()
    rewards = train_episodes.reward()
    dones = train_episodes.done()
    next_states = train_episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = advantages.to(log_probs.device)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)

def fast_adapt_a2c(clone, train_episodes, adapt_lr, baseline, gamma, tau, first_order=False):
    second_order = not first_order
    loss = maml_a2c_loss(train_episodes, clone, baseline, gamma, tau)
    gradients = autograd.grad(loss,
                              clone.parameters(),
                              retain_graph=second_order,
                              create_graph=second_order)
    return l2l.algorithms.maml.maml_update(clone, adapt_lr, gradients)

def meta_surrogate_loss(iteration_replays, iteration_policies, policy, baseline, tau, gamma, adapt_lr):
    mean_loss = 0.0
    mean_kl = 0.0
    for task_replays, old_policy in zip(iteration_replays, iteration_policies):
        train_replays = task_replays[:-1]
        valid_episodes = task_replays[-1]
        new_policy = l2l.clone_module(policy)

        for train_episodes in train_replays:
            new_policy = fast_adapt_a2c(new_policy, train_episodes, adapt_lr,
                                        baseline, gamma, tau, first_order=False)
        states, actions = valid_episodes.state(), valid_episodes.action()
        rewards, dones = valid_episodes.reward(), valid_episodes.done()
        next_states = valid_episodes.next_state()

        old_densities = old_policy.density(states)
        new_densities = new_policy.density(states)
        kl = kl_divergence(new_densities, old_densities).mean()
        mean_kl += kl

        advantages = compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states)
        advantages = ch.normalize(advantages).detach()
        old_log_probs = old_densities.log_prob(actions).mean(dim=1, keepdim=True).detach()
        new_log_probs = new_densities.log_prob(actions).mean(dim=1, keepdim=True)
        advantages = advantages.to(new_log_probs.device)
        mean_loss += trpo.policy_loss(new_log_probs, old_log_probs, advantages)
    return mean_loss / len(iteration_replays), mean_kl / len(iteration_replays)

# --- Core Trainer Class ---

class MAMLTRPOTrainer:
    def __init__(
        self,
        env_name='HalfCheetahForwardBackward-v1',
        adapt_lr=0.1,
        meta_lr=1.0,
        adapt_steps=1,
        meta_bsz=20,
        adapt_bsz=20,
        tau=1.00,
        gamma=0.95,
        seed=42,
        num_workers=10,
        cuda=True,
    ):
        self.env_name = env_name
        self.adapt_lr = adapt_lr
        self.meta_lr = meta_lr
        self.adapt_steps = adapt_steps
        self.meta_bsz = meta_bsz
        self.adapt_bsz = adapt_bsz
        self.tau = tau
        self.gamma = gamma
        self.seed = seed
        self.num_workers = num_workers
        self.cuda = cuda

        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)

        self.device = torch.device('cpu')
        if self.cuda and torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            self.device = torch.device('cuda')

        def make_env():
            env = gym.make(self.env_name)
            return ch.envs.ActionSpaceScaler(env)

        self.env = l2l.gym.AsyncVectorEnv([make_env for _ in range(self.num_workers)])
        self.env.seed(self.seed)
        self.env.set_task(self.env.sample_tasks(1)[0])
        self.env = ch.envs.Torch(self.env)
        
        self.policy = DiagNormalPolicy(self.env.state_size, self.env.action_size, device=self.device)
        self.baseline = LinearValue(self.env.state_size, self.env.action_size)
        if self.cuda:
            self.policy.to(self.device)
            self.baseline.to(self.device)

    def train(self, num_iterations=1000):
        for iteration in range(num_iterations):
            iteration_reward = 0.0
            iteration_replays = []
            iteration_policies = []

            for task_config in tqdm(self.env.sample_tasks(self.meta_bsz), leave=False, desc=f'Iteration {iteration}'):
                clone = deepcopy(self.policy)
                self.env.set_task(task_config)
                self.env.reset()
                task = ch.envs.Runner(self.env)
                task_replay = []

                for step in range(self.adapt_steps):
                    train_episodes = task.run(clone, episodes=self.adapt_bsz)
                    if self.cuda:
                        train_episodes.to(self.device, non_blocking=True)
                    clone = fast_adapt_a2c(clone, train_episodes, self.adapt_lr,
                                           self.baseline, self.gamma, self.tau, first_order=False)
                    task_replay.append(train_episodes)

                valid_episodes = task.run(clone, episodes=self.adapt_bsz)
                task_replay.append(valid_episodes)
                iteration_reward += valid_episodes.reward().sum().item() / self.adapt_bsz
                iteration_replays.append(task_replay)
                iteration_policies.append(clone)
            
            adaptation_reward = iteration_reward / self.meta_bsz
            yield {
                'iteration': iteration,
                'adaptation_reward': adaptation_reward,
            }

            self._meta_optimize(iteration_replays, iteration_policies)

    def _meta_optimize(self, iteration_replays, iteration_policies):
        backtrack_factor = 0.5
        ls_max_steps = 15
        max_kl = 0.01

        if self.cuda:
            self.policy.to(self.device, non_blocking=True)
            self.baseline.to(self.device, non_blocking=True)
            iteration_replays = [[r.to(self.device, non_blocking=True) for r in task_replays] for task_replays in iteration_replays]

        old_loss, old_kl = meta_surrogate_loss(iteration_replays, iteration_policies, self.policy, self.baseline, self.tau, self.gamma, self.adapt_lr)
        grad = autograd.grad(old_loss, self.policy.parameters(), retain_graph=True)
        grad = parameters_to_vector([g.detach() for g in grad])
        
        Fvp = trpo.hessian_vector_product(old_kl, self.policy.parameters())
        step = trpo.conjugate_gradient(Fvp, grad)
        shs = 0.5 * torch.dot(step, Fvp(step))
        lagrange_multiplier = torch.sqrt(shs / max_kl)
        step = step / lagrange_multiplier
        
        step_ = [torch.zeros_like(p.data) for p in self.policy.parameters()]
        vector_to_parameters(step, step_)
        step = step_
        old_loss.detach_()

        for ls_step in range(ls_max_steps):
            stepsize = backtrack_factor ** ls_step * self.meta_lr
            clone = deepcopy(self.policy)
            for p, u in zip(clone.parameters(), step):
                p.data.add_(-stepsize, u.data)
            
            new_loss, kl = meta_surrogate_loss(iteration_replays, iteration_policies, clone, self.baseline, self.tau, self.gamma, self.adapt_lr)
            if new_loss < old_loss and kl < max_kl:
                for p, u in zip(self.policy.parameters(), step):
                    p.data.add_(-stepsize, u.data)
                break
    
    def save_model(self, path: str):
        """Saves the meta-learner's state dictionary to a file."""
        # Ensure the directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        checkpoint = {
            'meta_learner_state_dict': self.meta_learner.state_dict(),
            'baseline_state_dict': self.baseline.state_dict(),
        }
        
        torch.save(checkpoint, path)
        print(f"Checkpoint saved successfully to {path}")

    def load_model(self, path: str):
        """Loads the meta-learner's state dictionary from a file."""
        if not os.path.exists(path):
            print(f"Warning: Model file not found at {path}. Starting from scratch.")
            return
        checkpoint = torch.load(path, map_location=self.device)
        
        self.meta_learner.load_state_dict(checkpoint['meta_learner_state_dict'])
        self.baseline.load_state_dict(checkpoint['baseline_state_dict'])

        print(f"Checkpoint loaded successfully from {path}")

if __name__ == '__main__':
    import wandb
    # This block allows running the script directly for a single experiment
    try:
        trainer = MAMLTRPOTrainer(
            env_name='HalfCheetahForwardBackward-v1',
            adapt_lr=0.008342,
            meta_lr=1.919,
            adapt_steps=5,
            meta_bsz=25,
            adapt_bsz=30,
            tau=0.9344,
            gamma=0.9087,
            seed=42,
            num_workers=10,
            cuda=True,
        )
        wandb.init()
        for metrics in trainer.train(num_iterations=600):
            wandb.log(metrics)
            print(
                f"Iteration {metrics['iteration'] + 1}: "
                f"Reward = {metrics['adaptation_reward']:.4f}, "
            )
        save_path = "model/maml_trpo_half.pth"
        trainer.save_model(save_path)
    except gym.error.DependencyNotInstalled:
        print("="*60)
        print("This example requires Mujoco. Please see the MAML-TRPO trainer for installation notes.")
        print("="*60)
    except Exception as e:
        print(f"An error occurred: {e}")