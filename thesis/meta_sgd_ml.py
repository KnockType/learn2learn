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

import learn2learn as l2l
import metaworld
from examples.rl.policies import DiagNormalPolicy

# --- Helper Functions (Identical to MAML-TRPO's inner loop loss) ---

def compute_advantages(baseline, tau, gamma, rewards, dones, states, next_states):
    """Computes the GAE advantages."""
    returns = ch.td.discount(gamma, rewards, dones)
    baseline.fit(states, returns)
    values = baseline(states)
    next_values = baseline(next_states)
    bootstraps = values * (1.0 - dones) + next_values * dones
    next_value = torch.zeros(1, device=values.device)
    return ch.pg.generalized_advantage(tau, gamma, rewards, dones, bootstraps, next_value)

def a2c_loss(episodes, learner, baseline, gamma, tau):
    """Computes the A2C loss for both inner and outer loops."""
    states = episodes.state()
    actions = episodes.action()
    rewards = episodes.reward()
    dones = episodes.done()
    next_states = episodes.next_state()
    log_probs = learner.log_prob(states, actions)
    advantages = compute_advantages(baseline, tau, gamma, rewards,
                                    dones, states, next_states)
    advantages = ch.normalize(advantages).detach()
    return a2c.policy_loss(log_probs, advantages)

# --- Environment Wrappers and Task Samplers (Copied from MAML-TRPO) ---

class _InfiniteSampler:
    """A helper class that creates an infinite iterator to yield random batches."""
    def __init__(self, items: List[Any], batch_size: int):
        self.items = items
        self.num_items = len(items)
        self.batch_size = batch_size
        self.base_batch = self.items * (self.batch_size // self.num_items)
        self.effective_batch_size = self.batch_size - len(self.base_batch)
        self.iterator = self._create_iterator()

    def _create_iterator(self):
        if self.effective_batch_size > 0:
            sampler = SubsetRandomSampler(range(self.num_items))
            batch_sampler = BatchSampler(sampler, batch_size=self.effective_batch_size, drop_last=True)
            return iter(batch_sampler)
        return None

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> List[Any]:
        if self.iterator is None:
            return self.base_batch.copy()
        try:
            indices = next(self.iterator)
        except StopIteration:
            self.iterator = self._create_iterator()
            indices = next(self.iterator)
        return [self.items[i] for i in indices] + self.base_batch

class BalancedTaskSampler:
    """Samples balanced batches of tasks from a benchmark (for ML10)."""
    def __init__(self, benchmark, batch_size: int, test: bool = False):
        classes = benchmark.test_classes if test else benchmark.train_classes
        tasks = benchmark.test_tasks if test else benchmark.train_tasks
        env_names = list(classes.keys())
        tasks_by_env = defaultdict(list)
        for task in tasks:
            tasks_by_env[task.env_name].append(task)
        self.env_type_sampler = _InfiniteSampler(env_names, batch_size)
        self.task_samplers = {name: _InfiniteSampler(tasks, 1) for name, tasks in tasks_by_env.items()}

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> List[Any]:
        sampled_env_types = next(self.env_type_sampler)
        random.shuffle(sampled_env_types)
        return [next(self.task_samplers[name])[0] for name in sampled_env_types]

class TaskSampler:
    """Samples random batches of tasks from a list (for ML1)."""
    def __init__(self, tasks: List[Any], batch_size: int):
        self.tasks = tasks
        self.batch_size = batch_size

    def __iter__(self) -> Iterator:
        return self

    def __next__(self) -> List[Any]:
        return [random.choice(self.tasks) for _ in range(self.batch_size)]

class MetaWorldBenchmarkEnv(l2l.gym.MetaEnv):
    """Unified wrapper for Meta-World benchmarks (ML1 and ML10)."""
    def __init__(self, benchmark, seed=None, test=False):
        self.benchmark = benchmark
        self.test = test
        self.tasks = benchmark.test_tasks if test else benchmark.train_tasks
        self.classes = benchmark.test_classes if test else benchmark.train_classes
        self._active_env = None
        self.set_task(self.tasks[0]) # Initialize with the first task

    @property
    def observation_space(self):
        return self._active_env.observation_space

    @property
    def action_space(self):
        return self._active_env.action_space

    def set_task(self, task):
        env_cls = self.classes[task.env_name]
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

# --- Meta-SGD Trainer Class ---

class MetaSGDTrainer:
    """
    A trainer class for the Meta-SGD algorithm with Advantage Actor-Critic (A2C) and Meta-World.
    """
    def __init__(
        self,
        benchmark_name: str, # 'ml1' or 'ml10'
        env_name: str = None,  # Required for ML1
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
        # Input Validation
        assert benchmark_name in ['ml1', 'ml10'], "Benchmark must be 'ml1' or 'ml10'."
        if benchmark_name == 'ml1' and env_name is None:
            raise ValueError("env_name must be specified for ML1 benchmark.")
        
        self.benchmark_name = benchmark_name
        self.env_name = env_name
        self.fast_lr_init = fast_lr_init
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

        # Create Benchmark and Task Sampler
        if self.benchmark_name == 'ml1':
            self.benchmark = metaworld.ML1(self.env_name, seed=self.seed)
            self.train_task_sampler = TaskSampler(self.benchmark.train_tasks, batch_size=self.meta_bsz)
        elif self.benchmark_name == 'ml10':
            self.benchmark = metaworld.ML10(seed=self.seed)
            self.train_task_sampler = BalancedTaskSampler(self.benchmark, batch_size=self.meta_bsz)
        
        # Create a dummy env to get observation and action shapes
        dummy_env = MetaWorldBenchmarkEnv(self.benchmark, seed=self.seed)
        state_size = dummy_env.observation_space.shape[0]
        action_size = dummy_env.action_space.shape[0]

        # Create AsyncVectorEnv
        def make_env():
            env = MetaWorldBenchmarkEnv(self.benchmark, seed=self.seed)
            env = ch.envs.ActionSpaceScaler(env)
            return env

        self.env = l2l.gym.AsyncVectorEnv([make_env for _ in range(self.num_workers)])
        self.env.seed(self.seed)
        self.env = ch.envs.Torch(self.env)
        
        # Initialize Policy and Baseline
        policy = DiagNormalPolicy(state_size, action_size, device=self.device)
        self.meta_learner = l2l.algorithms.MetaSGD(policy, lr=self.fast_lr_init, first_order=False)
        self.baseline = LinearValue(state_size, action_size).to(self.device)
        
        self.meta_learner.to(self.device)
        self.baseline.to(self.device)
        self.meta_optimizer = optim.Adam(self.meta_learner.parameters(), lr=self.meta_lr)

    def train(self, num_iterations: int = 300) -> Iterator[Dict[str, Any]]:
        """
        Starts the training process. Yields metrics at each iteration.
        """
        for iteration in range(num_iterations):
            self.meta_optimizer.zero_grad()
            iteration_reward = 0.0
            iteration_meta_loss = 0.0

            with tqdm(total=self.meta_bsz, desc=f'Iteration {iteration+1}/{num_iterations}', leave=False) as pbar:
                for task_i, task_config in enumerate(next(self.train_task_sampler)):
                    # 1. Create a clone of the meta-learner
                    learner = self.meta_learner.clone()
                    
                    # 2. Initialize and set task
                    self.env.set_task(task_config)
                    self.env.reset()
                    task = ch.envs.Runner(self.env)

                    # 3. Inner loop adaptation
                    for step in range(self.adapt_steps):
                        train_episodes = task.run(learner, episodes=self.adapt_bsz)
                        train_episodes = train_episodes.to(self.device)
                        inner_loss = a2c_loss(train_episodes, learner, self.baseline, self.gamma, self.tau)
                        learner.adapt(inner_loss)

                    # 4. Compute meta-loss on validation episodes
                    valid_episodes = task.run(learner, episodes=self.adapt_bsz)
                    valid_episodes = valid_episodes.to(self.device)
                    meta_loss = a2c_loss(valid_episodes, learner, self.baseline, self.gamma, self.tau)
                    
                    iteration_reward += valid_episodes.reward().sum().item() / self.adapt_bsz
                    iteration_meta_loss += meta_loss
                    pbar.update(1)

            # 5. Meta-optimization step
            adaptation_reward = iteration_reward / self.meta_bsz
            meta_loss = iteration_meta_loss / self.meta_bsz
            
            # Backpropagate the meta-loss and update the meta-learner
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

def fast_adapt_meta_sgd(clone, episodes, baseline, gamma, tau, first_order, device):
    """
    Performs a single adaptation step using the Meta-SGD update rule.
    
    It uses the learning rates stored within the `clone` model.
    """
    second_order = not first_order
    
    # Calculate loss on the adaptation data
    loss = a2c_loss(episodes, clone, baseline, gamma, tau, device)

    # Get gradients of the loss with respect to the policy's parameters
    # Assumes your Meta-SGD model stores weights in `clone.parameters()`
    params = list(clone.parameters())
    gradients = torch.autograd.grad(loss,
                                    params,
                                    retain_graph=second_order,
                                    create_graph=second_order)

    # Perform the Meta-SGD update: new_param = param - lr * grad
    # Assumes your model stores the learned learning rates in `clone.adapt_lrs`
    if not hasattr(clone, 'adapt_lrs'):
        raise AttributeError("The provided policy model does not have 'adapt_lrs'. "
                             "Ensure your Meta-SGD model defines them, e.g., as nn.ParameterList.")

    updated_params = []
    for param, grad, lr in zip(params, gradients, clone.adapt_lrs):
        updated_param = param - lr * grad
        updated_params.append(updated_param)

    # Create a new policy instance with the updated parameters
    # This requires your model to have a .clone() method that accepts new parameters.
    # If not, you might need to use a library like `higher` or manually update.
    return clone.clone(new_params=updated_params)

def evaluate_meta_sgd(benchmark, policy, baseline, gamma, tau, n_workers, seed, cuda):
    """
    Evaluates a Meta-SGD policy on a benchmark.
    
    The learning rates for adaptation are part of the 'policy' model itself.
    """
    device = torch.device('cuda' if cuda else 'cpu')

    # Parameters
    adapt_steps = 3
    adapt_bsz = 10  # Number of episodes to collect per adaptation step
    n_eval_tasks = 10

    tasks_reward = 0
    policy.to(device)
    baseline.to(device)

    if args.benchmark == 'ml1':
        test_benchmark = metaworld.ML1(args.env_name, seed=args.seed)
    else: # args.benchmark == 'ml10'
        test_benchmark = metaworld.ML10(seed=args.seed)
    test_env = MetaWorldBenchmarkEnv(test_benchmark, seed=args.seed, test=True)
    test_env = ch.envs.ActionSpaceScaler(test_env)

    n_eval_tasks = 10 # 10 task types
    if args.benchmark == 'ml10':
        task_sampler = BalancedTaskSampler(test_benchmark, batch_size=n_eval_tasks, test=True)
    else:
        task_sampler = TaskSampler(test_benchmark.test_tasks, batch_size=n_eval_tasks)

    env = ch.envs.Torch(test_env)
    task_sampler = BalancedTaskSampler(benchmark, batch_size=n_eval_tasks, test=True)
    
    results_by_class = defaultdict(list)
    
    # Iterate over a batch of test tasks
    for i, task in enumerate(next(task_sampler)):
        clone = deepcopy(policy)
        env.set_task(task)
        env.reset()
        task_run = ch.envs.Runner(env)

        # Adapt the policy for a few steps
        for step in range(adapt_steps):
            adapt_episodes = task_run.run(clone, episodes=adapt_bsz)
            if cuda:
                adapt_episodes = adapt_episodes.to(device, non_blocking=True)
            
            # Perform a Meta-SGD update step
            clone = fast_adapt_meta_sgd(clone,
                                        adapt_episodes,
                                        baseline,
                                        gamma,
                                        tau,
                                        first_order=True,
                                        device=device)

        # Evaluate the final adapted policy
        eval_episodes = task_run.run(clone, episodes=adapt_bsz)

        # Calculate and log metrics
        task_reward = eval_episodes.reward().sum().item() / adapt_bsz
        
        # Success rate calculation (assuming 'success' is in info dict)
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
            
        print(f"Task {i+1}/{n_eval_tasks} | Success Rate: {task_success_rate:.2f} | Reward: {task_reward:.2f}")
        results_by_class[f"{task.env_name}_success_rate"].append(task_success_rate)
        tasks_reward += task_reward

    final_eval_reward = tasks_reward / n_eval_tasks
    print(f"\nAverage reward over {n_eval_tasks} test tasks: {final_eval_reward:.3f}")

    # Log results to wandb if available
    if wandb.run:
        num_train_tasks = 5 # Example value, adjust as needed
        for i in range(min(len(next(iter(results_by_class.values()))), adapt_bsz // num_train_tasks)):
            res = {}
            for env_name, successes in results_by_class.items():
                res[env_name] = successes[i]
            wandb.log(res)
            
    return final_eval_reward

def evaluate(args, trainer):
    """Evaluates the Meta-SGD model on test tasks."""
    print("\n--- Evaluating on Test Tasks ---")
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    trainer.baseline.to(device)
    if args.benchmark == 'ml1':
        test_benchmark = metaworld.ML1(args.env_name, seed=args.seed)
    else: # args.benchmark == 'ml10'
        test_benchmark = metaworld.ML10(seed=args.seed)
    test_env = MetaWorldBenchmarkEnv(test_benchmark, seed=args.seed, test=True)
    test_env = ch.envs.ActionSpaceScaler(test_env)

    n_eval_tasks = 10 # 10 task types
    if args.benchmark == 'ml10':
        test_task_sampler = BalancedTaskSampler(test_benchmark, batch_size=n_eval_tasks, test=True)
    else:
        test_task_sampler = TaskSampler(test_benchmark.test_tasks, batch_size=n_eval_tasks)

    env = ch.envs.Torch(test_env)
    
    results_by_class = defaultdict(list)
    total_reward = 0

    with torch.no_grad():
        for task in tqdm(next(test_task_sampler), desc="Evaluation"):
            clone = deepcopy(trainer.meta_learner)
            clone.to(device)
            env.set_task(task)
            env.reset()
            task_runner = ch.envs.Runner(env)

            # Adapt
            for _ in range(args.adapt_steps):
                adapt_episodes = task_runner.run(clone, episodes=args.adapt_bsz)
                adapt_episodes.to(device)
                inner_loss = a2c_loss(adapt_episodes, clone, trainer.baseline, args.gamma, args.tau)
                clone.adapt(inner_loss)

            eval_episodes = task_runner.run(clone, episodes=args.adapt_bsz)
            successes = [info['success'] for info in eval_episodes.info()]
            success_rate = np.mean(successes)

            results_by_class[f"{task.env_name}_success_rate"].append(success_rate)
            total_reward += eval_episodes.reward().sum().item() / args.adapt_bsz

    avg_eval_reward = total_reward / n_eval_tasks
    log_dict = {'test_reward': avg_eval_reward}
    
    if args.benchmark == 'ml10':
        for env_name, successes in results_by_class.items():
            log_dict[env_name] = np.mean(successes)
    
    avg_success_rate = np.mean([s for succs in results_by_class.values() for s in succs])
    log_dict['test_success_rate_avg'] = avg_success_rate

    print(f"Average Test Reward: {avg_eval_reward:.4f}")
    print(f"Average Test Success Rate: {avg_success_rate:.4f}")
    wandb.log(log_dict)

def main(args):
    # Initialize Trainer
    trainer = MetaSGDTrainer(
        benchmark_name=args.benchmark,
        env_name=args.env_name,
        fast_lr_init = args.fast_lr_init,
        meta_lr=args.meta_lr,
        adapt_steps=args.adapt_steps,
        meta_bsz=args.meta_bsz,
        adapt_bsz=args.adapt_bsz,
        tau=args.tau,
        gamma=args.gamma,
        seed=args.seed,
        num_workers=args.num_workers,
        cuda=args.cuda,
    )

    # Initialize W&B
    wandb.init(
        project=f"l2l-metaworld-{args.benchmark}-{args.env_name or 'all'}-metasgd",
        config=vars(args)
    )

    # Load Checkpoint if it exists
    trainer.load_model("model/meta_sgd.pth")

    # Train and Evaluate
    for i, metrics in enumerate(trainer.train(num_iterations=args.num_iterations)):
        wandb.log(metrics)
        print(
            f"Iteration {metrics['iteration'] + 1}: "
            f"Reward = {metrics['adaptation_reward']:.4f}, "
            f"Meta Loss = {metrics['meta_loss']:.4f}"
        )
        
        if i % 25 == 0:
            evaluate(args, trainer) # Pass trainer instance

    # Save the final model
    save_path = f"model/meta_sgd_{args.benchmark}_{args.env_name or 'all'}.pth"
    trainer.save_model(save_path)

    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Meta-SGD on Meta-World")
    parser.add_argument('--benchmark', type=str, choices=['ml1', 'ml10'], required=True, help="Benchmark to use.")
    parser.add_argument('--env_name', type=str, default=None, help="Environment name for ML1 (e.g., 'reach-v2').")
    parser.add_argument('--fast_lr_init', type=float, default=0.1, help="Initial fast adaptation learning rate")
    # Hyperparameters
    parser.add_argument('--num_iterations', type=int, default=500, help="Number of meta-iterations.")
    parser.add_argument('--meta_lr', type=float, default=0.001, help="Outer loop learning rate (Adam).")
    parser.add_argument('--adapt_steps', type=int, default=1, help="Number of adaptation steps in the inner loop.")
    parser.add_argument('--meta_bsz', type=int, default=20, help="Meta-batch size (number of tasks).")
    parser.add_argument('--adapt_bsz', type=int, default=20, help="Adaptation batch size (episodes per task).")
    parser.add_argument('--gamma', type=float, default=0.99, help="Discount factor.")
    parser.add_argument('--tau', type=float, default=1.00, help="GAE parameter.")
    
    # General settings
    parser.add_argument('--seed', type=int, default=42, help="Random seed.")
    parser.add_argument('--num_workers', type=int, default=8, help="Number of parallel workers.")
    parser.add_argument('--cuda', type=int, default=1, help="Whether to use CUDA (1 for True, 0 for False).")
    
    args = parser.parse_args()
    main(args)