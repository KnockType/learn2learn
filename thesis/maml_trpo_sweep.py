#!/usr/bin/env python3
import os
import time
"""
A script to create and run a parallel Weights & Biases sweep
by loading the configuration from a YAML file.

This script:
1. Loads the sweep configuration from a specified YAML file.
2. Calls the wandb API to create the sweep and get its ID.
3. Spawns multiple processes, each running a wandb agent for the sweep.

Usage:
    python maml_trpo_sweep.py maml_trpo_sweep_config.yaml --project your_project_name --entity your_username
"""

import wandb
import yaml
import argparse
import multiprocessing
from maml_trpo_trainer import MAMLTRPOTrainer

def train_for_sweep():
    """
    This is the core training function that will be called by `wandb.agent`.
    It initializes a wandb run and then executes the training logic.
    """
    
    wandb.init(name=f"maml_sweep_{time.time()}")
    config = wandb.config

    trainer = MAMLTRPOTrainer(
        adapt_lr=config.adapt_lr,
        meta_lr=config.meta_lr,
        adapt_steps=config.adapt_steps,
        gamma=config.gamma,
        tau=config.tau,
        adapt_bsz=config.adapt_bsz,
        meta_bsz=config.meta_bsz,
    )

    for metrics in trainer.train(num_iterations=60):
        wandb.log(metrics)


def run_single_agent(sweep_id, project, entity, count, gpu_id):
    """
    A wrapper function to start a single wandb agent in a process.
    It sets the CUDA_VISIBLE_DEVICES environment variable.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    print(f"[Agent on GPU {gpu_id}] Starting for sweep {sweep_id} with a target of {count} runs...")
    try:
        wandb.agent(sweep_id, function=train_for_sweep, project=project, entity=entity, count=count)
    except Exception as e:
        print(f"[Agent on GPU {gpu_id}] Execution failed: {e}")
        
def main():
    parser = argparse.ArgumentParser(
        description="Launch a parallel wandb sweep from a YAML config file."
    )
    parser.add_argument(
        "--config_file",
        default="maml_trpo_sweep_config.yaml",
        type=str,
        help="Path to the YAML sweep configuration file."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=f"meta-rl-sweeps-{time.time()}",
        help="The wandb project name to use for the sweep."
    )
    parser.add_argument(
        "--entity",
        type=str,
        help="Your wandb username or team name."
    )
    parser.add_argument(
        "-n", "--num_agents",
        type=int,
        default=4,
        help="Number of parallel agents to launch."
    )
    parser.add_argument(
        "-r", "--total_runs",
        type=int,
        default=40,
        help="Total number of hyperparameter combinations to try across all agents."
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use for the agents (e.g., '0,1,2,3')."
    )
    args = parser.parse_args()

    gpu_ids = [int(gid) for gid in args.gpus.split(',')]
    num_gpus = len(gpu_ids)

    # 1. Load Sweep Configuration from YAML
    try:
        with open(args.config_file, 'r') as f:
            sweep_configuration = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_file}")
        return
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        return

    # 2. Create the Sweep on the wandb Server
    print("Creating the sweep from YAML file...")
    sweep_id = wandb.sweep(
        sweep=sweep_configuration,
        project=args.project,
        entity=args.entity,
    )
    print(f"Sweep created successfully! Sweep ID: {sweep_id}")
    print(f"Launching {args.num_agents} agents to run a total of {args.total_runs} trials.")

    # 3. Distribute runs and Launch Parallel Agents
    runs_per_agent = args.total_runs // args.num_agents
    extra_runs = args.total_runs % args.num_agents

    processes = []
    for i in range(args.num_agents):
        count = runs_per_agent + (1 if i < extra_runs else 0)
        if count > 0:
            gpu_id = gpu_ids[i % num_gpus]
            
            # Create a Process object for each agent
            process = multiprocessing.Process(
                target=run_single_agent,
                args=(sweep_id, args.project, args.entity, count, gpu_id)
            )
            
            processes.append(process)
            print(f"Launching agent {i+1}/{args.num_agents} on GPU {gpu_id}...")
            process.start()

    try:
        # Wait for all processes to complete their work
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("\nKeyboard interrupt received. Terminating all agent processes...")
        for process in processes:
            process.terminate()
            process.join()

    print("All sweep agents have finished.")

if __name__ == '__main__':
    main()