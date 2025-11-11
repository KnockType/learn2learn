# thesis/l2l_rl2_inner_loop.py

import random
import numpy as np
import torch
import gym
import learn2learn as l2l
from rl2_buffers import Lifetime_buffer
from rl2_agent import Outer_loop_action_agent

def run_inner_loop(arguments, training=True, run_deterministically=False):
    config, model_path, env_name, task = arguments
    is_metaworld = config.env_name in ["ML1", "ML10"]

    # -------- SETUP --------
    def make_env(env_name, task):
        env = gym.make(env_name)
        env.set_task(task)
        env = gym.wrappers.ClipAction(env)
        if config.seeding:
            env.seed(config.seed)
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)
        return env

    env = make_env(env_name, task)

    actions_size = env.action_space.shape[0]
    obs_size = env.observation_space.shape[0]

    meta_agent = Outer_loop_action_agent(
        actions_size=actions_size,
        obs_size=obs_size,
        rnn_input_size=config.rnn_input_size,
        rnn_type=config.rnn_type,
        rnn_hidden_state_size=config.rnn_hidden_state_size,
        initial_std=config.initial_std
    )
    meta_agent.load_state_dict(torch.load(model_path))

    if config.seeding:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.il_device == 'auto':
        il_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        il_device = config.il_device
    meta_agent = meta_agent.to(il_device)
    lifetime_buffer = Lifetime_buffer(config.num_il_lifetime_steps, env, il_device, env_name=env_name)

    # -------- Inner loop --------
    episode_step_num = 0
    max_episode_steps = 200

    episode_return = 0
    episodes_returns = []
    episodes_successes=[]  # it keeps track of wether the goal was achieved in each episode
    succeded_in_episode=False  #keeps track of wether the agent has achieved succes in current episode


    next_obs = env.reset()
    next_obs = torch.tensor(next_obs, dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)

    action = torch.zeros(env.action_space.shape[0])
    logprob = torch.zeros(1)
    reward = torch.zeros(1)


    hidden_state = meta_agent.initialize_state(batch_size=1)
    if isinstance(hidden_state, tuple):
        hidden_state = tuple(hs.to(il_device) for hs in hidden_state)
    else:
        hidden_state = hidden_state.to(il_device)


    # -- MAIN LOOP --
    for global_step in range(config.num_il_lifetime_steps):
        obs, prev_done = next_obs, done

        lifetime_buffer.store_step_data(global_step=global_step, obs=obs.to(il_device), prev_act=action.to(il_device),
                                         prev_reward=reward.to(il_device), prev_logp=logprob.to(il_device), prev_done=prev_done.to(il_device))

        hidden_state = meta_agent.rnn_next_state(lifetime_buffer, lifetime_timestep=global_step,
                                                 rnn_current_state=hidden_state)

        with torch.no_grad():
            if not run_deterministically:
                meta_value = meta_agent.get_value(hidden_state).squeeze(0).squeeze(0)
                action, logprob, _ = meta_agent.get_action(hidden_state)
                action = action.squeeze(0).squeeze(0)
                logprob = logprob.squeeze(0).squeeze(0)
            else: # For evaluation
                meta_value = torch.ones(1)
                action = meta_agent.get_deterministic_action(hidden_state)
                action = action.squeeze(0).squeeze(0)
                logprob = torch.zeros(1)

        lifetime_buffer.store_meta_value(global_step=global_step, meta_value=meta_value)

        # Execute action in environment
        next_obs, reward, done, info = env.step(action.cpu().numpy())
        done = torch.tensor(done).to(il_device)

        reward = torch.tensor(reward).to(il_device)
        next_obs = torch.tensor(next_obs).to(il_device)

        episode_step_num += 1
        episode_return += reward
        if is_metaworld and info['success'] == 1.0: 
            succeded_in_episode = True 


        if global_step == config.num_il_lifetime_steps - 1:
            dummy_obs = torch.zeros(env.observation_space.shape[0])
            lifetime_buffer.store_step_data(global_step=global_step + 1, obs=dummy_obs.to(il_device), prev_act=action.to(il_device),
                                             prev_reward=reward.to(il_device), prev_logp=logprob.to(il_device), prev_done=done.to(il_device))

        # Handle episode end
        if episode_step_num >= max_episode_steps or done:
            episodes_returns.append(episode_return)
            if is_metaworld:
                episodes_successes.append(succeded_in_episode)
            
            done = torch.ones(1).to(il_device)
            next_obs = env.reset()
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(il_device)
            episode_step_num = 0
            episode_return = 0
            succeded_in_episode=False

    lifetime_buffer.episodes_returns = episodes_returns
    if is_metaworld:
        lifetime_buffer.episodes_successes = episodes_successes 

    # In l2l envs, 'success' is not a standard metric, so we return only returns

    return lifetime_buffer if training else episodes_returns, episodes_successes