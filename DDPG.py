'''
@Author  ：Yan JP
@Created on Date：2023/6/4 11:00
'''
from matplotlib.font_manager import FontProperties  # 导入字体模块

import warnings

warnings.filterwarnings("ignore")
import numpy as np
import torch
import argparse
import datetime
import time
import gym
import random
import torch.optim as optim
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter

seed = 10
np.random.seed(seed)
torch.manual_seed(seed)


def chinese_font():
    try:
        font = FontProperties(
            # 系统字体路径
            fname='C:\\Windows\\Fonts\\方正粗黑宋简体.ttf', size=14)
    except:
        font = None
    return font


# Ornstein–Uhlenbeck噪声
'''
 在强化学习中，Ornstein–Uhlenbeck噪声通常被用作探索策略的一种方法，
 因为它可以产生相关的噪声序列，从而使得动作选择更加平滑和连续23。
 '''
import para


class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.4, min_sigma=0.1, decay_period=1000):  # 原本100000
        self.mu = mu  # OU噪声的参数
        self.theta = theta  # OU噪声的参数
        self.sigma = max_sigma  # OU噪声的参数
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period - 20
        self.n_actions = action_space
        self.low = -1
        # self.high = np.sqrt(para.maxPower)
        self.high = 1
        self.reset()
        self.eps = 0.99

    def reset(self):
        self.obs = np.ones(self.n_actions) * self.mu

    def evolve_obs(self):
        x = self.obs
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.n_actions)
        self.obs = x + dx
        return self.obs

    def get_action(self, action, t=0):
        ou_obs = self.evolve_obs()
        # self.sigma*=self.eps
        # self.sigma = max(self.min_sigma, self.sigma * self.eps)
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)  # sigma会逐渐衰减
        return np.clip(action + ou_obs, self.low, self.high)  # 动作加上噪声后进行剪切


# 经验回放对象
class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity  # 经验回放的容量
        self.buffer = []  # 缓冲区
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        ''' 缓冲区是一个队列，容量超出时去掉开始存入的转移(transition)
        '''
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)  # 随机采出小批量转移
        state, action, reward, next_state, done = zip(*batch)  # 解压成状态，动作等
        return state, action, reward, next_state, done

    def __len__(self):
        ''' 返回当前存储的量
        '''
        return len(self.buffer)


# 演员网络（给定状态，输出动作）
class Actor(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(n_states, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, n_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))
        # x = torch.sigmoid(self.linear3(x))
        return x


# 评论员网络（给定状态-动作对，做出评价）
class Critic(nn.Module):
    def __init__(self, n_states, n_actions, hidden_dim, init_w=3e-3):
        super(Critic, self).__init__()

        self.linear1 = nn.Linear(n_states + n_actions, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        # 随机初始化为较小的值
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        # 按维数1拼接
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


# 深度确定性策略梯度算法对象
class DDPG:
    def __init__(self, n_states, n_actions, arg_dict):
        self.var = arg_dict['var']
        self.device = torch.device(arg_dict['device'])
        # DDPG要训练四个网络：Q网络，Q-target网络，策略网络，策略-target网络
        self.critic = Critic(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)
        self.actor = Actor(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)
        self.target_critic = Critic(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)
        self.target_actor = Actor(n_states, n_actions, arg_dict['hidden_dim']).to(self.device)

        # 复制参数到目标网络
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=arg_dict['critic_lr'])
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=arg_dict['actor_lr'])
        self.memory = ReplayBuffer(arg_dict['memory_capacity'])
        self.batch_size = arg_dict['batch_size']
        self.soft_tau = arg_dict['soft_tau']  # 软更新参数
        self.gamma = arg_dict['gamma']

    def choose_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        return action.detach().cpu().numpy()[0]

    def update(self):
        if len(self.memory) < self.batch_size:  # 当 memory 中不满足一个批量时，不更新策略
            return
        # self.var =max(0.2,self.var* 0.9995)
        # 从经验回放中(replay memory)中随机采样一个批量的转移(transition)
        state, action, reward, next_state, done = self.memory.sample(self.batch_size)
        # 转变为张量
        state = torch.FloatTensor(np.array(state)).to(self.device)
        next_state = torch.FloatTensor(np.array(next_state)).to(self.device)
        action = torch.FloatTensor(np.array(action)).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policy_loss = self.critic(state, self.actor(state))
        policy_loss = -policy_loss.mean()
        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        # expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = reward + self.gamma * target_value

        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()
        # 软更新
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.soft_tau) +
                param.data * self.soft_tau
            )

    def save_model(self, path):

        # torch.save(self.actor.state_dict(), path + 'upper_agent_UE{}.pt'.format(para.K))
        torch.save(self.actor.state_dict(), path + 'agent_upper_beam.pt'.format(para.K))

    def load_model(self, path):

        self.actor.load_state_dict(torch.load(path))
