'''
@Author  ：Yan JP
@Created on Date：2023/5/6 15:12
'''
import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Dueling_Net(nn.Module):
    def __init__(self, args):
        super(Dueling_Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        if args.use_noisy:
            pass
            # self.V = NoisyLinear(args.hidden_dim, 1)
            # self.A = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.V = nn.Linear(args.hidden_dim, 1)
            self.A = nn.Linear(args.hidden_dim, args.action_dim)

    def forward(self, s):
        s = torch.relu(self.fc1(s))
        s = torch.relu(self.fc2(s))
        V = self.V(s)  # batch_size X 1
        A = self.A(s)  # batch_size X action_dim
        Q = V + (A - torch.mean(A, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_dim).to(device=device)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim).to(device=device)
        if args.use_noisy:
            pass
            # self.fc3 = NoisyLinear(args.hidden_dim, args.action_dim)
        else:
            self.fc3 = nn.Linear(args.hidden_dim, args.action_dim).to(device=device)

    def forward(self, s):
        s = s.to(device)
        s = torch.relu(self.fc1(s)).to(device=device)
        s = torch.relu(self.fc2(s)).to(device=device)
        # s = F.softmax(self.fc1(s),dim=1).to(device=device)
        # s = F.softmax(self.fc2(s),dim=1).to(device=device)
        Q = self.fc3(s).to(device=device)
        return Q.to(device=device)

class DQN(object):
    def __init__(self, args):
        self.action_dim = args.action_dim
        self.batch_size = args.batch_size  # batch size
        self.max_train_steps = args.max_train_steps
        self.lr = args.lr  # learning rate
        self.gamma = args.gamma  # discount factor
        self.tau = args.tau  # Soft update
        self.use_soft_update = args.use_soft_update
        self.target_update_freq = args.target_update_freq  # hard update
        self.update_count = 0

        self.grad_clip = args.grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_double = args.use_double
        self.use_dueling = args.use_dueling
        self.use_per = args.use_per
        self.use_n_steps = args.use_n_steps
        if self.use_n_steps:
            self.gamma = self.gamma ** args.n_steps

        if self.use_dueling:  # Whether to use the 'dueling network'
            self.net = Dueling_Net(args).to(device=device)
        else:
            self.net = Net(args).to(device=device)

        self.target_net = copy.deepcopy(self.net)  # Copy the online_net to the target_net

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        for param in self.net.parameters():
            param.to(device)
    def choose_action(self, state, epsilon):
        with torch.no_grad():
            state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0)
            q = self.net(state)
            if np.random.uniform() > epsilon:
                action = q.argmax(dim=-1).item()
            else:
                action = np.random.randint(0, self.action_dim)
            return action

    def learn(self, replay_buffer, total_steps):
        batch, batch_index, IS_weight = replay_buffer.sample(total_steps)

        with torch.no_grad():  # q_target has no gradient
            if self.use_double:  # Whether to use the 'double q-learning' True
                # Use online_net to select the action
                a_argmax = self.net(batch['next_state']).argmax(dim=-1, keepdim=True)  # shape：(batch_size,1)
                # Use target_net to estimate the q_target
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * self.target_net(
                    batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)
                # q_target = batch['reward'] + self.gamma * self.target_net(batch['next_state']).gather(-1, a_argmax).squeeze(-1)  # shape：(batch_size,)

            else:
                q_target = batch['reward'] + self.gamma * (1 - batch['terminal']) * \
                           self.target_net(batch['next_state']).max(dim=-1)[0]  # shape：(batch_size,)

        q_current = self.net(batch['state']).gather(-1, batch['action']).squeeze(-1)  # shape：(batch_size,)
        td_errors = q_current - q_target  # shape：(batch_size,)

        loss = (td_errors ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.grad_clip:
            torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
        self.optimizer.step()

        if self.use_soft_update:  # soft update
            for param, target_param in zip(self.net.parameters(), self.target_net.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:  # hard update
            self.update_count += 1
            if self.update_count % self.target_update_freq == 0:
                self.target_net.load_state_dict(self.net.state_dict())

        if self.use_lr_decay:  # learning rate Decay
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_now = 0.9 * self.lr * (1 - total_steps / self.max_train_steps) + 0.1 * self.lr
        for p in self.optimizer.param_groups:
            p['lr'] = lr_now
