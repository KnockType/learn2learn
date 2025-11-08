import imageio
import random
from copy import deepcopy
import os

import cherry as ch
import gym
import gymnasium
import numpy as np
import torch
from cherry.algorithms import a2c, trpo
from cherry.models.robotics import LinearValue
from torch import autograd
from torch.distributions.kl import kl_divergence
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm import tqdm
from maml_trpo_trainer import MAMLTRPOTrainer

def evaluate_and_record(env, policy, filename="eval_episode.mp4"):

    frames = []

    obs = env.reset()

    done, truncated = False, False

    while not (done or truncated):

        frame = env.render()

        frames.append(frame)

        action = policy(obs)

        _ = env.step(action)

    #imageio.mimsave(filename, frames, fps=30)


envname = "RampPush-v0"
seed = 42

trainer = MAMLTRPOTrainer(
            env_name=envname,
            adapt_lr=0.1,
            meta_lr=1.0,
            adapt_steps=2,
            meta_bsz=20,
            adapt_bsz=20,
            tau=1.0,
            gamma=0.95,
            seed=seed,
            num_workers=10,
            cuda=True,
        )

trainer.load_model(f"/home/jonwee/l2l/meta_rl/thesis/model/maml_trpo_{envname}_{seed}_iter{500}.pth")

def make_env():
    env = gym.make(envname)
    return ch.envs.ActionSpaceScaler(env)

env = make_env()

env.seed(seed)
env.set_task(env.sample_tasks(1)[0])
env = ch.envs.Torch(env)

evaluate_and_record(env, trainer.policy)
