#!/usr/bin/env python3
"""
A script to create and run a parallel Weights & Biases sweep for Meta-SGD
by loading the configuration from a YAML file.
"""

import os
import yaml
import argparse
import multiprocessing
import wandb
from meta_sgd_trainer import MetaSGDTrainer # IMPORT THE NEW TRAINER

def train_for_sweep():
    """
    Core training function called by `wandb.agent`.
    """
    wandb.init()
    config = wandb.config

    # INSTANTIATE THE NEW TRAINER
    trainer = MetaSGDTrainer(
        env_name=config.env_name,
        fast_lr_init=config.fast_lr_init,
        meta_lr=config.meta_lr,
        adapt_steps=config.adapt_steps,
        gamma=config.gamma,
        tau=config.tau,
        adapt_bsz=config.adapt_bsz,
        meta_bsz=config.meta_bsz,
    )

    for metrics in trainer.train(num_iterations=config.num_iterations):
        wandb.log(metrics)


def run_single_agent(sweep_id: str, project: str, entity: str, count: int, gpu_id: int):
    """
    Wrapper function to start a single wandb agent in a dedicated process.
    """
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    print(f"[Agent on GPU {gpu_id}] Starting for sweep '{sweep_id}' with a target of {count} runs...")
    try:
        wandb.agent(sweep_id, function=train_for_sweep, project=project, entity=entity, count=count)
    except Exception as e:
        print(f"[Agent on GPU {gpu_id}] Execution failed: {e}")
        
def main():
    parser = argparse.ArgumentParser(description="Launch a parallel wandb sweep for Meta-SGD.")
    parser.add_argument(
        "--config_file",
        default="meta_sgd_sweep_config.yaml", # Default to new config
        type=str,
        help="Path to the YAML sweep configuration file."
    )
    parser.add_argument(
        "--project",
        type=str,
        default="meta-sgd-sweep",
        help="The wandb project name to use for the sweep."
    )
    parser.add_argument("--entity", type=str, help="Your wandb username or team name.")
    parser.add_argument("-n", "--num_agents", type=int, default=4, help="Number of parallel agents to launch.")
    parser.add_argument("-r", "--total_runs", type=int, default=15, help="Total trials to run.")
    parser.add_argument("--gpus", type=str, default="0", help="Comma-separated list of GPU IDs.")
    args = parser.parse_args()

    gpu_ids = [int(gid) for gid in args.gpus.split(',')]
    num_gpus = len(gpu_ids)

    try:
        with open(args.config_file, 'r') as f:
            sweep_configuration = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {args.config_file}")
        return

    print("Creating the sweep on wandb server...")
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=args.project, entity=args.entity)
    print(f"Sweep created successfully! Sweep ID: {sweep_id}")
    print(f"View sweep at: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}")

    runs_per_agent = args.total_runs // args.num_agents
    extra_runs = args.total_runs % args.num_agents
    processes = []
    for i in range(args.num_agents):
        count = runs_per_agent + (1 if i < extra_runs else 0)
        if count > 0:
            gpu_id = gpu_ids[i%num_gpus]
            process = multiprocessing.Process(
                target=run_single_agent,
                args=(sweep_id, args.project, args.entity, count, gpu_id)
            )
            processes.append(process)
            print(f"Launching agent {i+1}/{args.num_agents} on GPU {gpu_id} to run {count} trials...")
            process.start()

    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        print("\nInterrupt received. Terminating agent processes...")
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()

    print("All sweep agents have finished.")

if __name__ == '__main__':
    main()