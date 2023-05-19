'''
@Author  ：Yan JP
@Created on Date：2023/5/18 23:11 
'''
'''
@Author  ：Yan JP
@Created on Date：2023/5/5 22:51 
'''
import torch
import numpy as np
import os
import argparse
import datetime
from para import *
import env
import baseline
import pickle
from dqn import DQN
from replay_buffer import N_Steps_Prioritized_ReplayBuffer
from tiles import tile
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class CustomUnpickler(pickle.Unpickler):

    def find_class(self, module, name):
        try:
            return super().find_class(__name__, name)
        except AttributeError:
            return super().find_class(module, name)


class Runner():
    def __init__(self, args, ttile: tile, fov_id):
        self.args = args
        # self.seed = seed
        # np.random.seed(seed)
        # torch.manual_seed(seed)

        self.env = env.subEnvironment(1, ttile, fov_id)
        self.args.state_dim = self.env.state_dim
        self.args.action_dim = self.env.action_dim

        # current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        current_time = datetime.datetime.now().strftime("%Y%m%d")

        self.train_log_dir = 'runs/DQN/' + current_time
        self.writer = SummaryWriter(log_dir=self.train_log_dir)
        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.total_steps = 0  # Record the total steps during the training
        self.epsilon = self.args.epsilon_init
        self.epsilon_min = self.args.epsilon_min
        self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

        self.replay_buffer = N_Steps_Prioritized_ReplayBuffer(args)

        with open('runs/model/agent.pkl', 'rb') as f:
            agent = pickle.load(f)

        self.agent = agent

        self.algorithm = 'DQN'
        if args.use_double:
            self.algorithm += '_Double'
        if args.use_dueling:
            self.algorithm += '_Dueling'
        if args.use_noisy:
            self.algorithm += '_Noisy'
        if args.use_per:
            self.algorithm += '_PER'
        if args.use_n_steps:
            self.algorithm += "_N_steps"
        # self.writer = SummaryWriter(log_dir='runs/DQN_{}/seed_{}'.format(self.algorithm,  seed))

        self.evaluate_num = 0  # Record the number of evaluations
        self.evaluate_rewards = []  # Record the rewards during the evaluating
        self.train_rewards = []  # Record the rewards during the evaluating

        self.total_steps = 0  # Record the total steps during the training
        if args.use_noisy:  # 如果使用Noisy net，就不需要epsilon贪心策略了
            self.epsilon = 0
        else:
            self.epsilon = self.args.epsilon_init
            self.epsilon_min = self.args.epsilon_min
            self.epsilon_decay = (self.args.epsilon_init - self.args.epsilon_min) / self.args.epsilon_decay_steps

    def evaluate_online(self, ):

        evaluate_reward = 0
        # self.agent.net.eval()  # 模型不会更新参数，也不会使用一些只在训练时有效的层，例如dropout或batch normalization。这样可以提高模型的预测性能和稳定性。
        r = 0
        for i in range(1):
            res = []
            state = self.env.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            while not done:
                action = self.agent.choose_action(state, epsilon=0)
                res.append(action)
                episode_steps += 1
                next_state, reward, done, _ = self.env.step(action, episode_steps)
                episode_reward += reward
                state = next_state
            # evaluate_reward += episode_reward
            # self.agent.net.train()
            # self.evaluate_rewards.append(evaluate_reward)
            # print("###########     {}     ###########\n".format(evaluate_reward))
            r = episode_reward
            print("-----------------total_steps:{} \t evaluate_reward:{} \t epsilon：{}".format(self.total_steps,
                                                                                               episode_reward,
                                                                                               self.epsilon))
            # 统计结果
            new_res = []
            for a in res:
                if a < 5:
                    k = 1  # 1表示压缩
                else:
                    k = 0
                l = a  # 范围就是0-9
                new_res.append([k, l])
            self.new_res = new_res
            print(new_res)
            self.env.get_info()
        return r

    def greedy(self, fov_id):
        self.baseline = baseline.greedyMethod(ttile, fov_id)
        self.baseline.reset()
        for index in range(0, N_F):
            self.baseline.step(index)
        self.baseline.get_info()
        pass


import time

if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    # parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    parser.add_argument("--max_train_steps", type=int, default=int(1.2e4), help=" Maximum number of training steps")

    parser.add_argument("--evaluate_freq", type=float, default=2e3,
                        help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--evaluate_times", type=float, default=1, help="Evaluate times")

    parser.add_argument("--buffer_capacity", type=int, default=int(0.6e5),
                        help="The maximum replay-buffer capacity ")  # 原本1e5
    parser.add_argument("--batch_size", type=int, default=256, help="batch size")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of actor")
    parser.add_argument("--gamma", type=float, default=0.90, help="Discount factor")
    parser.add_argument("--epsilon_init", type=float, default=0.5, help="Initial epsilon")
    parser.add_argument("--epsilon_min", type=float, default=0.1, help="Minimum epsilon")
    parser.add_argument("--epsilon_decay_steps", type=int, default=int(0.1e5),
                        help="How many steps before the epsilon decays to the minimum")  # 原本1e5
    parser.add_argument("--tau", type=float, default=0.005, help="soft update the target network")
    parser.add_argument("--use_soft_update", type=bool, default=True, help="Whether to use soft update")
    parser.add_argument("--target_update_freq", type=int, default=200,
                        help="Update frequency of the target network(hard update)")
    parser.add_argument("--n_steps", type=int, default=4, help="n_steps")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Learning rate Decay")
    parser.add_argument("--grad_clip", type=float, default=0, help="Gradient clip")  # 原本10.0

    parser.add_argument("--use_double", type=bool, default=True, help="Whether to use double Q-learning")
    parser.add_argument("--use_dueling", type=bool, default=False, help="Whether to use dueling network")
    parser.add_argument("--use_noisy", type=bool, default=False, help="Whether to use noisy network")
    parser.add_argument("--use_per", type=bool, default=True, help="Whether to use PER")
    parser.add_argument("--use_n_steps", type=bool, default=True, help="Whether to use n_steps Q-learning")

    args = parser.parse_args()
    # seed = 3

    ttile = CustomUnpickler(open('tiles.pkl', 'rb')).load()

    writer2 = SummaryWriter(log_dir='runs/evaluate')

    # for i in range(30):
    #     fov_id = np.random.randint(0,N_fovs)
    #     print(fov_id)
    #     runner = Runner(args=args, ttile=ttile, fov_id=fov_id)
    #     # runner.run()
    #
    #     episode_reward=runner.evaluate_online()
    #     writer2.add_scalar('evaluate_rewards:', episode_reward, global_step=i+1)

    runner = Runner(args=args, ttile=ttile, fov_id=3)
    runner.greedy(fov_id=3)
