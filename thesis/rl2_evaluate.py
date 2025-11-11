# thesis/evaluate_l2l_rl2.py

import os
import time
import argparse
import random
import numpy as np
import torch
import learn2learn as l2l
import gymnasium as gym # Use gymnasium for the RecordVideo wrapper

from .rl2_agent import Outer_loop_action_agent
from .rl2_config import get_config

def evaluate_and_record(args):
    # --- Config and Initialization ---
    config = get_config('l2l_rl2')
    config.env_name = args.env_name # Override env_name from args

    # --- Setup ---
    if config.seeding:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Video Directory ---
    run_name = os.path.basename(args.model_path).replace('__best_model.pth', '')
    video_dir = f"videos/{run_name}_eval_{int(time.time())}"
    os.makedirs(video_dir, exist_ok=True)
    print(f"Saving videos to: {video_dir}")

    # --- Load Model ---
    dummy_env = l2l.gym.make_env(config.env_name) # For obs/action sizes
    actions_size = dummy_env.action_space.shape[0]
    obs_size = dummy_env.observation_space.shape[0]

    meta_agent = Outer_loop_action_agent(
        actions_size=actions_size,
        obs_size=obs_size,
        rnn_input_size=config.rnn_input_size,
        rnn_type=config.rnn_type,
        rnn_hidden_state_size=config.rnn_hidden_state_size,
        initial_std=config.initial_std,
    ).to(device)

    meta_agent.load_state_dict(torch.load(args.model_path, map_location=device))
    meta_agent.eval() # Set to evaluation mode

    # --- Evaluation Loop ---
    env_for_tasks = l2l.gym.make_env(config.env_name)
    tasks = env_for_tasks.sample_tasks(args.num_tasks)
    all_task_returns = []

    for i, task in enumerate(tasks):
        print(f"Running evaluation for task {i+1}/{args.num_tasks}")
        
        # Create and wrap the environment for recording
        eval_env = l2l.gym.make_env(config.env_name)
        eval_env.set_task(task)
        # Record the first episode of the agent's lifetime in this task
        record_env = gym.wrappers.RecordVideo(
            eval_env,
            video_folder=os.path.join(video_dir, f"task_{i}"),
            episode_trigger=lambda x: x == 0, # Record only the first episode
            name_prefix=f"rl2-eval-{config.env_name}"
        )

        # --- Run one lifetime (inner loop rollout) ---
        with torch.no_grad():
            episode_return = 0
            episodes_returns = []
            
            next_obs, _ = record_env.reset()
            next_obs = torch.from_numpy(next_obs).float().to(device)
            hidden_state = meta_agent.initialize_state(batch_size=1)
            if isinstance(hidden_state, tuple):
                hidden_state = tuple(hs.to(device) for hs in hidden_state)
            else:
                hidden_state = hidden_state.to(device)
            
            # Create a dummy buffer just for the rnn_next_state method
            from l2l_rl2_buffers import Lifetime_buffer
            dummy_buffer = Lifetime_buffer(config.num_il_lifetime_steps, dummy_env, device)
            
            for step in range(config.num_il_lifetime_steps):
                obs = next_obs
                
                # Update dummy buffer to get RNN input
                prev_action = torch.zeros(actions_size) if step == 0 else action
                prev_reward = torch.zeros(1) if step == 0 else reward
                prev_done = torch.zeros(1) if step == 0 else done
                dummy_buffer.store_step_data(step, obs, prev_action, prev_reward, torch.zeros(1), prev_done)
                
                hidden_state = meta_agent.rnn_next_state(dummy_buffer, lifetime_timestep=step, rnn_current_state=hidden_state)
                action = meta_agent.get_deterministic_action(hidden_state).squeeze()

                # Environment step
                next_obs, reward, terminated, truncated, info = record_env.step(action.cpu().numpy())
                done = torch.max(torch.Tensor([terminated, truncated]))
                next_obs = torch.from_numpy(next_obs).float().to(device)
                episode_return += reward

                if terminated or truncated:
                    episodes_returns.append(episode_return)
                    episode_return = 0
                    next_obs, _ = record_env.reset()
                    next_obs = torch.from_numpy(next_obs).float().to(device)
            
            if not episodes_returns: # If no episode finished
                episodes_returns.append(episode_return)
        
        all_task_returns.append(np.mean(episodes_returns))
        record_env.close()

    print("\n--- Evaluation Summary ---")
    print(f"Average return over {args.num_tasks} tasks: {np.mean(all_task_returns):.2f} +/- {np.std(all_task_returns):.2f}")
    print(f"Videos saved in '{video_dir}' directory.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a trained RL^2 agent and record videos.')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the saved .pth model file.')
    parser.add_argument('--env-name', type=str, default='HalfCheetahForwardBackward-v1', help='Name of the learn2learn environment.')
    parser.add_argument('--num-tasks', type=int, default=5, help='Number of evaluation tasks to run.')
    parser.add_argument('--seed', type=int, default=123, help='Random seed for evaluation.')
    args = parser.parse_args()
    evaluate_and_record(args)