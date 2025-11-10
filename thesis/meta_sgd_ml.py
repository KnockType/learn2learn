#!/usr/bin/env python3

"""
Core implementation of the Meta-SGD algorithm with A2C, encapsulated in a reusable class,
and with support for both Meta-World ML1 and ML10 benchmarks.
"""
from copy import deepcopy
import random
import os
import argparse
from typing import Iterator, Dict, Any, List
import wandb

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c
from cherry.models.robotics import LinearValue
from torch import optim
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import SubsetRandomSampler, BatchSampler
from moviepy import ImageSequenceClip

import learn2learn as l2l
import metaworld
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
    
class MetaWorldML1(l2l.gym.MetaEnv):
    """
    Wrapper for the Meta-World ML1 benchmark to make it compatible
    with the learn2learn MetaEnv interface.
    """
    def __init__(self, env_name: str, seed=None, test=False, render_mode="rgb_array"):
        self.test = test
        self.ml1 = metaworld.ML1(env_name, seed=seed)
        self.render_mode = render_mode
        
        # Select the correct task list
        tasks = self.ml1.test_tasks if test else self.ml1.train_tasks
        self.tasks = tasks
        
        self._active_env = None
        # Set an initial task to define observation and action spaces
        self.set_task(tasks[0])
       
    @property
    def observation_space(self):
        return self._active_env.observation_space

    @property
    def action_space(self):
        return self._active_env.action_space

    def set_task(self, task):
        """Sets the active environment based on the task description."""
        env_name = task.env_name
        env_cls = self.ml1.test_classes[env_name] if self.test else self.ml1.train_classes[env_name]
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

    def render(self, mode="rgb_array", **kwargs):
        self._active_env.render_mode = mode
        return self._active_env.render(**kwargs)



class MetaWorldML10(l2l.gym.MetaEnv):
    """
    Wrapper for the Meta-World ML10 benchmark to make it compatible
    with the learn2learn MetaEnv interface.
    """
    def __init__(self, seed=None, test=False, render_mode="rgb_array"):
        self.test = test
        self.render_mode = render_mode
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

    def render(self, mode="rgb_array", **kwargs):
        self._active_env.render_mode = mode
        return self._active_env.render()

env_name = "door-close-v3"

def make_env(bench="ML1", seed=42, test=False, render_mode=None):
    def fn():
        if bench=="ML1":
            env = MetaWorldML1(env_name, seed=seed, test=test, render_mode=render_mode)
        else:
            env = MetaWorldML10(seed=seed, test=test, render_mode=render_mode)
        env = ch.envs.ActionSpaceScaler(env)
        return env
    return fn

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
        bench: str = 'ML10',
        fast_lr_init: float = 0.1,
        meta_lr: float = 0.001,
        adapt_steps: int = 1,
        meta_bsz: int = 20,
        adapt_bsz: int = 20,
        tau: float = 1.00,
        gamma: float = 0.99,
        seed: int = 42,
        num_workers: int = 10,
        cuda: bool = True,
    ):
        self.bench = bench
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

        self.benchmark = metaworld.ML10(seed=seed) if bench=="ML10" else metaworld.ML1(env_name, seed=seed)
        self.task_sampler = BalancedTaskSampler(self.benchmark, batch_size=meta_bsz)

        dummy_task = next(self.task_sampler)[0]
        dummy_env = self.benchmark.train_classes[dummy_task.env_name]()
        dummy_env.set_task(dummy_task)
        state_size = dummy_env.observation_space.shape[0]
        action_size = dummy_env.action_space.shape[0]


        self.env = l2l.gym.AsyncVectorEnv([make_env(bench=bench, seed=seed) for _ in range(self.num_workers)])
        self.env.seed(self.seed)
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

            tasks = next(self.task_sampler)
            for task_config in tqdm(tasks, leave=False, desc='Data Collection'):
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

            if iteration % 30 == 0:
                self.evaluate(iteration)

            yield {
                'iteration': iteration,
                'adaptation_reward': adaptation_reward,
                'meta_loss': meta_loss.item(),
            }

    def evaluate(self, iteration: int):
        """
        Evaluates the meta-learner on a set of test tasks.
        """
        print(f"\n--- Evaluating at Iteration {iteration} ---")
        adapt_steps = 3
        adapt_bsz = 10
        n_eval_tasks = 10
        total_reward = 0.0
        
        env = make_env(bench=self.bench, seed=self.seed, test=True)()
        env = ch.envs.Torch(env)
        task_sampler = BalancedTaskSampler(self.benchmark, batch_size=n_eval_tasks, test=True)
        results_by_class = defaultdict(list)
        video_frames = {}

        for task in tqdm(next(task_sampler), leave=False, desc="Evaluation"):
            learner = self.meta_learner.clone()
            learner = learner.to(self.device)
            env.set_task(task)
            env.reset()
            task_runner = ch.envs.Runner(env)
            frames = []

            # Adapt the policy
            for step in range(adapt_steps):
                adapt_episodes = task_runner.run(learner, episodes=adapt_bsz)
                adapt_episodes = adapt_episodes.to(self.device)
                inner_loss = maml_a2c_loss(adapt_episodes, learner, self.baseline, self.gamma, self.tau)
                learner.adapt(inner_loss)

            # Collect video frames
            with torch.no_grad():
                obs = env.reset()
                done = False
                while not done:
                    frame = env.unwrapped.render()
                    frames.append(frame)
                    action = learner(obs.to(self.device))
                    obs, _, done, _ = env.step(action)
            video_frames[task.env_name] = frames

            # Calculate evaluation reward
            eval_episodes = task_runner.run(learner, episodes=adapt_bsz)
            task_reward = eval_episodes.reward().sum().item() / adapt_bsz
            total_reward += task_reward

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

        # Save videos locally
        video_dir = f"videos/metasgd_{self.bench}_seed_{self.seed}"
        os.makedirs(video_dir, exist_ok=True)
        print(f"\nSaving evaluation videos to {video_dir}...")
        for task_name, frames in video_frames.items():
            if frames:
                try:
                    clip = ImageSequenceClip(frames, fps=15)
                    video_path = os.path.join(video_dir, f"task_{task_name}_iter_{iteration}.mp4")
                    clip.write_videofile(video_path, codec="libx264", logger=None)
                    print(f"  -> Saved {video_path}")
                except Exception as e:
                    print(f"  -> Could not save video for task {task_name}. Error: {e}")

        # Log metrics
        avg_reward = total_reward / n_eval_tasks
        print(f"Average evaluation reward: {avg_reward:.4f}\n")
        wandb.log({'evaluation_reward': avg_reward, 'iteration': iteration})
        num_train_tasks = 5
        for i in range(adapt_bsz//num_train_tasks):
            res = {}
            for env_name, successes in results_by_class.items():
                res[env_name] = successes[i]
            wandb.log(res)
        return avg_reward

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
        os.environ['CUDA_VISIBLE_DEVICES'] = str(5)
        seed = 1
        envname = 'ML1'
        trainer = MetaSGDTrainer(
            bench=envname,
            fast_lr_init = 0.08139,
            meta_lr = 0.004335,
            adapt_steps = 1,
            meta_bsz = 10,
            adapt_bsz = 20,
            tau = 1.00,
            gamma = 0.962,
            seed = seed,
            num_workers = 10,
            cuda = True,
        )
        wandb.init(project=f"meta_sgd_{envname}_", name=f"seed_{seed}")
        for metrics in trainer.train(num_iterations=500):
            wandb.log(metrics)
            print(
                f"Iteration {metrics['iteration'] + 1}: "
                f"Reward = {metrics['adaptation_reward']:.4f}, "
                f"Meta Loss = {metrics['meta_loss']:.4f}"
            )
        save_path = f"model/meta_sgd_{envname}_{seed}.pth"
        trainer.save_model(save_path)
    except gym.error.DependencyNotInstalled:
        print("="*60)
        print("This example requires Mujoco. Please see the MAML-TRPO trainer for installation notes.")
        print("="*60)
    except Exception as e:
        print(f"An error occurred: {e}")