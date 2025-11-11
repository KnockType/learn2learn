# thesis/train_l2l_rl2.py

import time
import numpy as np
import torch
import torch.optim as optim
import random
import wandb
import ray
import learn2learn as l2l
import gym
import os

from rl2_agent import Outer_loop_action_agent, Outer_loop_TBPTT_PPO
from rl2_buffers import OL_buffer, Lifetime_buffer
from rl2_utils import Statistics_tracker, Logger
from rl2_config import get_config
from rl2_inner_loop import run_inner_loop


def record_evaluation_video(agent, config, update_num, video_dir_base):
    """Records one lifetime of the agent and saves it."""
    print(f"\n--- Recording evaluation video at update {update_num} ---")

    video_dir = os.path.join(video_dir_base, f"update_{update_num}")
    
    # Create and wrap the environment
    env = gym.make(config.env_name)
    task = env.sample_tasks(1)[0]
    env.set_task(task)
    
    record_env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_dir,
        episode_trigger=lambda x: x == 0,
        name_prefix=f"rl2-train-{config.env_name}"
    )
    device = torch.device(config.il_device)
    with torch.no_grad():
        next_obs = record_env.reset()
        next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
        hidden_state = agent.initialize_state(batch_size=1)
        if isinstance(hidden_state, tuple):
            hidden_state = tuple(hs.to(device) for hs in hidden_state)
        else:
            hidden_state = hidden_state.to(device)
        done = torch.zeros(1).to(device)
        action=torch.from_numpy(np.zeros(env.action_space.shape[0])).to(device) #(have to do this because there is no action at timestep -1)
        logprob= torch.zeros(1).to(device)
        reward=torch.zeros(1).to(device)# pass 'reward' of 0 for timestep -1


        dummy_buffer = Lifetime_buffer(config.num_il_lifetime_steps, env, config.il_device)

        for step in range(config.num_il_lifetime_steps):
            obs, prev_done = next_obs, done
            dummy_buffer.store_step_data(step, obs, action, reward, logprob, prev_done)
            hidden_state = agent.rnn_next_state(dummy_buffer, lifetime_timestep=step, rnn_current_state=hidden_state)
            meta_value=agent.get_value(hidden_state).squeeze(0).squeeze(0) #(1)
            action, logprob, _ = agent.get_action(hidden_state)
            action=action.squeeze(0).squeeze(0)  #(action_size)
            logprob=logprob.squeeze(0).squeeze(0) #(1)

            next_obs, _, done, _ = record_env.step(action.cpu().numpy())
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            done = torch.tensor(done).to(device)
            if done:
                break # Stop after one episode
    
    record_env.close()
    print(f"--- Video saved to {video_dir} ---")


os.environ['CUDA_VISIBLE_DEVICES'] = str(3)
# --- Config and Initialization ---
config_setting = 'l2l_rl2'
config = get_config(config_setting)
is_metaworld = config.env_name in ["ML1", "ML10"]

if ray.is_initialized:
    ray.shutdown()
ray.init()

# --- Setup ---
# Create a dummy env to get action/obs sizes
dummy_env = gym.make(config.env_name)

# Seeding
if config.seeding:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    dummy_env.seed(config.seed)

# Device
if config.ol_device=='auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device=config.ol_device

# Agent
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

# Optimizer and Meta-Optimizer
optimizer = optim.Adam(meta_agent.parameters(), lr=config.learning_rate, eps=config.adam_eps)
TBPTT_PPO = Outer_loop_TBPTT_PPO(optimizer, logging=True, **config.ppo)

# Buffers and Utils
meta_buffer = OL_buffer(device=device)
data_statistics = Statistics_tracker()

logger = Logger(num_epsiodes_of_validation=config.num_epsiodes_of_validation, is_metaworld=is_metaworld)

# Logging
model_id = int(time.time())
run_name = f"RL2_seed_{config.seed}"
wandb.init(project=f"RL2_{config.env_name}", name=run_name, config=vars(config))

training_video_dir = f"videos/{run_name}_training"

# Model saving paths
model_path = f"./rl2/model/model_{config.env_name}_{config.seed}.pth"
best_model_path = f"./rl2/model/{run_name}__best_model.pth"
best_model_performance = -float('inf')

# Ray remote function
remote_inner_loop = ray.remote(run_inner_loop)

# --- Main Training Loop ---
print(f"Starting training for {config.env_name}")
start_time = time.time()

def validation_performance(logger):
    if not is_metaworld:
        performance = np.array(logger.validation_episodes_return[-config.num_lifetimes_for_validation:]).mean()
    else:
        performance= np.array(logger.validation_episodes_success_percentage[-config.num_lifetimes_for_validation:]).mean()
    return performance


for update_number in range(config.num_outer_loop_updates + 1):
    # --- Data Collection ---
    # Create a new env for this update to sample tasks from
    env = gym.make(config.env_name)
    tasks = env.sample_tasks(config.num_inner_loops_per_update)

    meta_agent = meta_agent.to(config.il_device)
    torch.save(meta_agent.state_dict(), model_path)

    # Prepare arguments for parallel workers
    inputs = [(config, model_path, config.env_name, task) for task in tasks]
    
    # Run inner loops in parallel
    lifetimes_buffers = ray.get([remote_inner_loop.options(num_cpus=1).remote(i) for i in inputs])
    
    # Process and aggregate data
    for lifetime_data in lifetimes_buffers:
        data_statistics.update_statistics(lifetime_data)
    
    for lifetime_data in lifetimes_buffers:
        if is_metaworld:
            lifetime_data.preprocess_data(data_stats=data_statistics, objective_mean=config.rewards_target_mean)
        
        lifetime_data.compute_meta_advantages_and_returns_to_go(gamma=config.meta_gamma, e_lambda=config.bootstrapping_lambda)
        logger.collect_per_lifetime_metrics(lifetime_data)
        meta_buffer.collect_lifetime_data(lifetime_data)

    meta_buffer.combine_data()

    # --- Logging ---
    logger.log_per_update_metrics(num_inner_loops_per_update=config.num_inner_loops_per_update)
    print(f'Update {update_number}: Mean episode return: {np.array(logger.lifetimes_mean_episode_return[-config.num_inner_loops_per_update:]).mean():.2f}')
    
    if update_number % 50 == 0:
        # We need the agent on the inner-loop device for the rollout
        meta_agent = meta_agent.to(config.il_device) 
        record_evaluation_video(meta_agent, config, update_number, training_video_dir)
        meta_agent = meta_agent.to(device) # Move it back to the outer-loop device


    # --- Save Best Model ---
    model_performance = validation_performance(logger)
    if model_performance > best_model_performance:
        best_model_performance = model_performance
        print(f'New best performance = {best_model_performance:.2f}. Saving model.')
        torch.save(meta_agent.state_dict(), best_model_path)

    # --- Update Model ---
    meta_agent = meta_agent.to(device)
    TBPTT_PPO.update(meta_agent=meta_agent, buffer=meta_buffer)
    meta_buffer.clean_buffer()

print('Training completed.')
print(f'Model ID = {model_id}')
print(f'Total time taken: {(time.time() - start_time) / 60:.2f} minutes')
wandb.finish()