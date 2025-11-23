import random
import numpy as np
import torch
import learn2learn as l2l
from rl2_buffers import Lifetime_buffer
from rl2_agent import Outer_loop_action_agent

import gym

def run_inner_loop(arguments , training=True, run_deterministically=False):

    config = arguments[0]
    model_path= arguments[1]
    benchmark_name= arguments[2]
    task= arguments[3]


    ########---------- SETUP ----------#########

    def create_env(task):
        env = gym.make(benchmark_name)
        env.unwrapped.set_task(task)  
        #env = gym.wrappers.ClipAction(env)
        if config.seeding:
            env.action_space.seed(config.seed)
            env.observation_space.seed(config.seed)
        return env
        
    #create env and load meta agent
    env= create_env(task)    
    actions_size=env.action_space.shape[0]
    obs_size=env.observation_space.shape[0] 

    meta_agent=Outer_loop_action_agent(       
                actions_size,          
                obs_size,         
                rnn_input_size=config.rnn_input_size,    
                rnn_type= config.rnn_type,              
                rnn_hidden_state_size= config.rnn_hidden_state_size,
                initial_std=config.initial_std 
                )
    meta_agent.load_state_dict(torch.load(model_path))

    #----
    if config.seeding==True:
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

    if config.il_device=='auto':
        il_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        il_device=torch.device(config.il_device)

    meta_agent=meta_agent.to(il_device)
    lifetime_buffer=Lifetime_buffer(config.num_il_lifetime_steps , env, il_device, env_name=f'{task["env_name"]}')
    
    ########-----------------------------###########


    #######--------- Inner loop ---------#############
    
    episode_step_num=0
    max_episode_steps=200 

    episode_return=0
    episodes_lengths=[]
    episodes_returns=[]


    #get an initial state from the agent to start training
    next_obs = torch.tensor(env.reset(),dtype=torch.float32).to(il_device)
    done = torch.zeros(1).to(il_device)


    ###-- MAIN LOOP --

    for global_step in range(0, config.num_il_lifetime_steps):

        #prepare for new step: next_obs is the new step obs , the done flag which indicates wether the episode finished in the last step becomes this episodes prev_done.
        obs, prev_done = next_obs, done

        #create dummy elements to store in the first step
        if global_step==0:
            action=torch.from_numpy(np.zeros(env.action_space.shape[0])) #(have to do this because there is no action at timestep -1)
            logprob= torch.zeros(1)
            reward=torch.zeros(1)# pass 'reward' of 0 for timestep -1

        #store (s_t ,a_{t-1} , logprob_{t-1}, d_t ,r_{t-1})
        lifetime_buffer.store_step_data(global_step=global_step, obs=obs.to(il_device), prev_act=action.to(il_device),
            prev_reward=reward.to(il_device), prev_logp=logprob.to(il_device),prev_done=prev_done.to(il_device))

        #-------OUTER LOOP agent involvement ; take action with RL2 agent ---------------
        # get meta agent predictions and state value estimates conditioning on the lifetime history
        if global_step==0:
            hidden_state=meta_agent.initialize_state(batch_size=1)
            if isinstance(hidden_state, tuple):
                hidden_state = tuple(hs.to(il_device) for hs in hidden_state)
            else:
                hidden_state.to(il_device)
            
        hidden_state=meta_agent.rnn_next_state(lifetime_buffer ,lifetime_timestep=global_step ,
                                            rnn_current_state=hidden_state)  #(1,1,hidden_state_size)
        
        if run_deterministically==False:
            with torch.no_grad():
                meta_value=meta_agent.get_value(hidden_state).squeeze(0).squeeze(0) #(1)
                action, logprob, _ = meta_agent.get_action(hidden_state)
                action=action.squeeze(0).squeeze(0)  #(action_size)
                logprob=logprob.squeeze(0).squeeze(0) #(1)

        #have meta agent take only deterministic actions. Should only be used in an evaluation setting (when not training it) .
        elif run_deterministically==True:
            with torch.no_grad():
                meta_value=torch.ones(1) #(1)
                action = meta_agent.get_deterministic_action(hidden_state)
                action=action.squeeze(0).squeeze(0)  #(action_size)
                logprob=torch.zeros(1) #(1)


        lifetime_buffer.store_meta_value(global_step=global_step, meta_value=meta_value)

        #-----------------------------------------------

        #execute the action and get environment response.
        next_obs, reward, done, info = env.step(action.cpu().numpy())
  
        #prepare for next step
        reward= torch.tensor(reward).to(il_device)
        done = torch.tensor(done).to(il_device)
        next_obs=torch.Tensor(next_obs).to(il_device)
        
        episode_step_num+=1 
        episode_return+= reward

        #When in last step save the last action taken and reward received in a extra timeslot
        if global_step== config.num_il_lifetime_steps-1:
            dummy_obs= torch.from_numpy(np.zeros(env.observation_space.shape[0])) #isnt used for anything ,could also save obs of timestep T+1 instead of zeros
            lifetime_buffer.store_step_data(global_step=global_step+1, obs=dummy_obs.to(il_device), prev_act=action.to(il_device),
            prev_reward=reward.to(il_device), prev_logp=logprob.to(il_device),prev_done=done.to(il_device))
            

        #deal with the case where the episode ends (note that meta world doesnt return terminated or truncated)
        if episode_step_num==max_episode_steps:
            #log metrics
            episodes_returns.append(episode_return)
            episodes_lengths.append(episode_step_num)

            #Prepare for next episode
            done=torch.ones(1).to(il_device) 
            next_obs = torch.tensor(env.reset(),dtype=torch.float32).to(il_device)
            episode_step_num=0
            episode_return=0




    lifetime_buffer.episodes_returns=episodes_returns

    #if its run for evaluation there is no need to return all the lifetime's data
    if training==False:
        return episodes_returns
    elif training==True:
        return lifetime_buffer