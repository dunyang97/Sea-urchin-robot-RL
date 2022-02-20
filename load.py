import gym
import pybullet_envs
import pybullet_envs.bullet as bul
import numpy as np
import torch
from sac_agent import soft_actor_critic_agent
from replay_memory import ReplayMemory

seed = 0
env = gym.make('AntBulletEnv-v0')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 0.0001
eval = True  ##
start_steps = 10000  ## Steps sampling random actions
replay_size = 1000000  ## size of replay buffer
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

agent = soft_actor_critic_agent(env.observation_space.shape[0], env.action_space, device=device, hidden_size=256,
                                seed=seed,
                                lr=LEARNING_RATE, gamma=0.99, tau=0.005, alpha=0.2)

agent.policy.load_state_dict(torch.load('dir_ant_lr0001-sc2500/weights_actor_final_1560.55'))
agent.critic.load_state_dict(torch.load('dir_ant_lr0001-sc2500/weights_critic_final_1560.55'))
