#!/usr/bin/env python3
"""
Core implementation of the Meta-SGD algorithm with A2C, encapsulated in a reusable class.
This script can be run directly for a single experiment.
"""
import random
import os
from typing import Iterator, Dict, Any

import cherry as ch
import gym
import numpy as np
import torch
import time
from cherry.algorithms import a2c
from cherry.models.robotics import LinearValue
from torch import optim
from tqdm import tqdm

import learn2learn as l2l
from examples.rl.policies import DiagNormalPolicy

def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    # Update baseline
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
    # Update policy and baseline
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


# --- Core Trainer Class ---

class MetaSGDTrainer:
    """
    A trainer class for the Meta-SGD algorithm with Advantage Actor-Critic (A2C).

    Args:
        env_name (str): The name of the Gym environment to use.
        fast_lr_init (float): The initial value for the per-parameter fast adaptation learning rates.
        meta_lr (float): Learning rate for the outer loop meta-optimizer (Adam).
        adapt_steps (int): Number of adaptation steps in the inner loop.
        meta_bsz (int): Number of tasks to sample per meta-iteration.
        adapt_bsz (int): Number of episodes to sample per adaptation step.
        tau (float): GAE discount factor.
        gamma (float): TD discount factor.
        seed (int): Random seed for reproducibility.
        num_workers (int): Number of parallel workers for environment sampling.
        cuda (bool): Whether to use GPU (if available).
    """
    def __init__(
        self,
        env_name: str = 'AntDirection-v1',
        fast_lr_init: float = 0.1,
        meta_lr: float = 0.001,
        adapt_steps: int = 1,
        meta_bsz: int = 20,
        adapt_bsz: int = 20,
        tau: float = 1.00,
        gamma: float = 0.99,
        seed: int = 42,
        num_workers: int = 8,
        cuda: bool = True,
    ):
        self.env_name = env_name
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

        # Create a dummy env to get observation and action shapes
        def make_env():
            env = gym.make(self.env_name)
            return ch.envs.ActionSpaceScaler(env)

        dummy_env = make_env()
        state_size = dummy_env.observation_space.shape[0]
        action_size = dummy_env.action_space.shape[0]
        dummy_env.close()

        self.env = l2l.gym.AsyncVectorEnv([make_env for _ in range(self.num_workers)])
        self.env.seed(self.seed)
        self.env.set_task(self.env.sample_tasks(1)[0])
        self.env = ch.envs.Torch(self.env)
        
        policy = DiagNormalPolicy(state_size, action_size, device=self.device)
        # Key difference: Wrap the model with MetaSGD
        self.meta_learner = l2l.algorithms.MetaSGD(policy, lr=fast_lr_init, first_order=False)
        self.baseline = LinearValue(state_size, action_size)
        
        self.meta_learner = self.meta_learner.to(self.device)
        self.baseline = self.baseline.to(self.device)

        # Key difference: Use a standard optimizer for the meta-parameters
        self.meta_optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.meta_lr)

    def train(self, num_iterations: int = 300) -> Iterator[Dict[str, Any]]:
        """
        Starts the training process. Yields metrics at each iteration.
        """
        for iteration in range(num_iterations):
            iteration_reward = 0.0
            iteration_meta_loss = 0.0

            for task_config in tqdm(self.env.sample_tasks(self.meta_bsz)):
                learner = self.meta_learner.clone()
                learner = learner.to(self.device)
                self.env.set_task(task_config)
                self.env.reset()
                task = ch.envs.Runner(self.env)
                    
                for step in range(self.adapt_steps):
                    train_episodes = task.run(learner, episodes=self.adapt_bsz)
                    train_episodes = train_episodes.to(self.device)
                    inner_loss = maml_a2c_loss(train_episodes, learner, self.baseline, self.gamma, self.tau)
                    learner.adapt(inner_loss)

                valid_episodes = task.run(learner, episodes=self.adapt_bsz)
                valid_episodes = valid_episodes.to(self.device)
                meta_loss = maml_a2c_loss(valid_episodes, learner, self.baseline, self.gamma, self.tau)
                    
                iteration_meta_loss += meta_loss
                iteration_reward += valid_episodes.reward().sum().item() / self.adapt_bsz

            # 5. Meta-optimization step
            adaptation_reward = iteration_reward / self.meta_bsz
            meta_loss = iteration_meta_loss / self.meta_bsz
            
            # Backpropagate the meta-loss and update the meta-learner
            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            yield {
                'iteration': iteration,
                'adaptation_reward': adaptation_reward,
                'meta_loss': meta_loss.item(),
            }

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
        os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
        seed = 42
        envname = 'RampPush-v0'
        trainer = MetaSGDTrainer(
            env_name=envname,
            fast_lr_init = 0.08,
            meta_lr = 0.004,
            adapt_steps = 1,
            meta_bsz = 10,
            adapt_bsz = 20,
            tau = 1.00,
            gamma = 0.962,
            seed = seed,
            num_workers = 10,
            cuda = True,
        )
        curr_time = time.time()
        save_interval = 250
        wandb.init(project=f"meta_sgd_{envname}_reward", name=f"seed_{seed}")
        for metrics in trainer.train(num_iterations=500):
            wandb.log(metrics)
            print(
                f"Iteration {metrics['iteration'] + 1}: "
                f"Reward = {metrics['adaptation_reward']:.4f}, "
                f"Meta Loss = {metrics['meta_loss']:.4f}"
            )
            current_iteration = metrics['iteration'] + 1
            if current_iteration % save_interval == 0:
                save_path = f"model/meta_sgd_{envname}_reward_{seed}_iter{current_iteration}_{curr_time}.pth"
                trainer.save_model(save_path)
        save_path = f"model/meta_sgd_{envname}_reward_{seed}.pth"
        trainer.save_model(save_path)
    except gym.error.DependencyNotInstalled:
        print("="*60)
        print("This example requires Mujoco. Please see the MAML-TRPO trainer for installation notes.")
        print("="*60)
    except Exception as e:
        print(f"An error occurred: {e}")