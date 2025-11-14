
import time
import numpy as np
import torch
import torch.optim as optim
import gymnasium as gym

import learn2learn as l2l
import random
#import sys
#sys.path.append('../')

from rl2_agent import Outer_loop_action_agent, Outer_loop_TBPTT_PPO 
from rl2_buffers import OL_buffer
from rl2_utils import Statistics_tracker , Sampler , Tasks_batch_sampler
from rl2_config import get_config
from rl2_inner_loop import run_inner_loop
from rl2_utils import Logger

import random
import os
from typing import Iterator, Dict, Any

import cherry as ch
import gym
import numpy as np
import torch
from cherry.algorithms import a2c
from cherry.models.robotics import LinearValue
from torch import optim
from tqdm import tqdm

import learn2learn as l2l
from examples.rl.policies import DiagNormalPolicy



import wandb
import ray
import os

os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

if ray.is_initialized:
    ray.shutdown()
ray.init()

load_weights=False #wether to initialize the meta agents with pretrained weights
initial_weights_path= 'path_to_model/model_name.pth'

config_setting='metaworld'
config=get_config(config_setting)
benchmark_name= config.benchmark_name

example_env = gym.make(benchmark_name)

#Initialize WyB and save config.
model_id=int(time.time())
run_name = f"RL2_{config.seed}__{model_id}"
wandb.init(project=f"rl2_{config.benchmark_name}",
                name= run_name,
                config=vars(config))



#Set performance metric for determining when the model achieves best performance (used for determining when to save the model) 
def validation_performance(logger):
    performance= np.array(logger.validation_episodes_return[-config.num_lifetimes_for_validation:]).mean()
    return performance



#--------Settings-----------------------
if config.seeding==True:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

if config.ol_device=='auto':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
else:
    device=torch.device(config.ol_device)

actions_size=example_env.action_space.shape[0]
obs_size=example_env.observation_space.shape[0] 


meta_agent=Outer_loop_action_agent(       
                actions_size,          
                obs_size,         
                rnn_input_size=config.rnn_input_size ,    
                rnn_type= config.rnn_type,              
                rnn_hidden_state_size= config.rnn_hidden_state_size,
                initial_std=config.initial_std ,
                ).to(device)

if load_weights==True:
    meta_agent.load_state_dict(torch.load(initial_weights_path))


optimizer = optim.Adam(meta_agent.parameters(), lr=config.learning_rate, eps=config.adam_eps)

meta_buffer=OL_buffer(device=device)


TBPTT_PPO=Outer_loop_TBPTT_PPO(optimizer, logging=True ,
                 k=config.ppo['k'], 
                 update_epochs=config.ppo['update_epochs'],
                 num_minibatches=config.ppo['num_minibatches'],
                 normalize_advantage=config.ppo['normalize_advantage'],
                 entropy_coef=config.ppo['entropy_coef'],
                 valuef_coef=config.ppo['valuef_coef'],
                 clip_grad_norm=config.ppo['clip_grad_norm'],
                 max_grad_norm=config.ppo['max_grad_norm'],
                 target_KL=config.ppo['target_KL'],
                 clip_coef=config.ppo['clip_coef'])


data_statistics=Statistics_tracker()

logger=Logger(num_epsiodes_of_validation=config.num_epsiodes_of_validation)

model_path = f"/home/jonwee/l2l/meta_rl/thesis/rl2_models/model_{config.benchmark_name}_{config.seed}.pth"
best_model_path= f"/home/jonwee/l2l/meta_rl/thesis/rl2_models/{config.benchmark_name}_{config.seed}__best_model.pth"
best_model_performance = 0 

remote_inner_loop=ray.remote(run_inner_loop) #allows for the inner loops to run on several cores paralelly
wandb.watch(meta_agent , log="all" ,log_freq=100) #to see gradients of the weights as histograms in the UI.


#------------------------------------------------------------------------------
#-------------------------------TRAIN OUTER LOOP AGENT ------------------------
start_time=time.time()

for update_number in range(config.num_outer_loop_updates+1):

    #----------------- DATA COLLECTION -----------------------
    #collect data from inner loops

    meta_agent = meta_agent.to(config.il_device)
    torch.save(meta_agent.state_dict(), model_path)

    inputs=[ (config,model_path,benchmark_name,task) for task in example_env.unwrapped.sample_tasks(config.num_inner_loops_per_update)]

    lifetimes_buffers = ray.get([remote_inner_loop.options(num_cpus=1).remote(i) for i in inputs]) 
    for lifetime_data in lifetimes_buffers:
        data_statistics.update_statistics(lifetime_data)
    for lifetime_data in lifetimes_buffers:
        lifetime_data.preprocess_data(data_stats=data_statistics , objective_mean=config.rewards_target_mean) #normalize rewards
        lifetime_data.compute_meta_advantages_and_returns_to_go(gamma=config.meta_gamma, e_lambda=config.bootstrapping_lambda)
        logger.collect_per_lifetime_metrics(lifetime_data)

        meta_buffer.collect_lifetime_data(lifetime_data)


    meta_buffer.combine_data()

    #----log some metrics to wandb and also print a few ------
    logger.log_per_update_metrics(num_inner_loops_per_update=config.num_inner_loops_per_update)
    #print(f'succes percentage at update {update_number} : {np.array(logger.lifetimes_success_percentage[-config.num_inner_loops_per_update:]).mean()}')
    print(f'mean episode return at update {update_number} : {np.array(logger.lifetimes_mean_episode_return[-config.num_inner_loops_per_update:]).mean()} ' )
    
    #---SAVE model parameters if it is the best yet -----------------
    model_performance=validation_performance(logger)
    if model_performance > best_model_performance:
        best_model_performance=model_performance
        print(f'new best performance={best_model_performance}')
        torch.save(meta_agent.state_dict(), best_model_path)

    # --------------  UPDATING MODEL ------------------------
    meta_agent=meta_agent.to(device)
    TBPTT_PPO.update(meta_agent=meta_agent,buffer=meta_buffer)

    meta_buffer.clean_buffer()

print('completed')
print(f'model_id= {model_id}')
print(f'time take in minutes ={(time.time()-start_time)/60}')
wandb.finish()

