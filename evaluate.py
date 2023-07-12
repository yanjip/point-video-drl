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
import Draw_pic
import para

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
        self.env = env.subEnvironment(1, ttile, fov_id)
        self.env2 = env.env_uncompress(1, ttile, fov_id)
        self.args.state_dim = self.env.state_dim
        self.args.action_dim = self.env.action_dim

        with open('runs/model/agent_choose_7_7.pkl', 'rb') as f:
            agent = pickle.load(f)

        self.agent = agent

        # self.writer = SummaryWriter(log_dir='runs/DQN_{}/seed_{}'.format(self.algorithm,  seed))
        with open('runs/model/agent_choose_uncompress.pkl', 'rb') as f:
            agent2 = pickle.load(f)
        self.agent2 = agent2

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
            print("-----------------\t evaluate_reward:{} \t".format(episode_reward))
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
            qoe = self.env.get_info()
        return qoe

    def evaluate_uncompress(self, ):
        r = 0
        for i in range(1):
            res = []
            state = self.env2.reset()
            done = False
            episode_reward = 0
            episode_steps = 0
            while not done:
                action = self.agent2.choose_action(state, epsilon=0)
                res.append(action)
                episode_steps += 1
                next_state, reward, done, _ = self.env2.step(action, episode_steps)
                episode_reward += reward
                state = next_state
            # evaluate_reward += episode_reward
            # self.agent.net.train()
            # self.evaluate_rewards.append(evaluate_reward)
            # print("###########     {}     ###########\n".format(evaluate_reward))
            r = episode_reward
            print("\n \t------baseline:uncompress-------")
            # 统计结果
            print(res)
            qoe = self.env2.get_info()
        return qoe

    def greedy(self, fov_id):
        self.baseline = baseline.greedyMethod(ttile, fov_id)
        self.baseline.reset()
        for index in range(0, N_F):
            self.baseline.step(index)
        qoe = self.baseline.get_info()
        return qoe
        pass

    def coarsness(self, fov_id):
        self.baseline_coarsness = baseline.tileCoarsness(ttile, fov_id)
        self.baseline_coarsness.reset()
        self.baseline_coarsness.step()

        qoe = self.baseline_coarsness.get_info()
        return qoe


# if __name__ == '__main__':
def once_test():
    # 测试加载训练好的agent
    proposed = []
    uncompress = []
    greedy = []
    coarsness = []
    fov_ids = []
    times = 30
    for i in range(times):
        fov_id = np.random.randint(0, N_fovs)
        fov_ids.append(fov_id)
        print("\nfov_id:", fov_id)
        runner = Runner(args=args, ttile=ttile, fov_id=fov_id)
        # runner.run()

        qoe1 = runner.evaluate_online()
        qoe2 = runner.evaluate_uncompress()
        # writer2.add_scalar('evaluate_rewards:', episode_reward, global_step=i + 1)
        qoe3 = runner.greedy(fov_id)
        qoe4 = runner.coarsness(fov_id)
        proposed.append(qoe1)
        uncompress.append(qoe2)
        greedy.append(qoe3)
        coarsness.append(qoe4)
    # Draw_pic.draw_evaluate(fov_ids, proposed, uncompress, greedy, coarsness)
    return sum(proposed) / times, sum(uncompress) / times, sum(greedy) / times, sum(coarsness) / times
    # 测试baseline-greedy
    # runner = Runner(args=args, ttile=ttile, fov_id=3)
    # runner.greedy(fov_id=2)
    #
    # # 测试baseline-greedy
    # runner = Runner(args=args, ttile=ttile, fov_id=2)
    # runner.coarsness(fov_id=2)


def qualityVsT():
    proposed = []
    uncompress = []
    greedy = []
    coarsness = []
    fov_ids = []
    ts = [0.1, 0.125, 0.150, 0.175, 0.20]
    for t in ts:
        para.T_slot = t
        p, u, g, c = once_test()
        proposed.append(p)
        greedy.append(g)
        coarsness.append(c)
        uncompress.append(u)
    Draw_pic.plot_QoE_t(ts, proposed, uncompress, greedy, coarsness)


def qualityVsF():
    proposed = []
    uncompress = []
    greedy = []
    coarsness = []
    fov_ids = []
    Fs = np.array([0.1e8, 0.5e8, 1.0e8, 1.5e8, 2.0e8])
    for F in Fs:
        para.F_max = F
        p, u, g, c = once_test()
        proposed.append(p)
        greedy.append(g)
        coarsness.append(c)
        uncompress.append(u)
    Draw_pic.plot_QoE_F(Fs / 1e9, proposed, uncompress, greedy, coarsness)


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    parser = argparse.ArgumentParser("Hyperparameter Setting for DQN")
    # parser.add_argument("--max_train_steps", type=int, default=int(4e5), help=" Maximum number of training steps")
    args = parser.parse_args()
    ttile = CustomUnpickler(open('tiles.pkl', 'rb')).load()

    # qualityVsT()

    qualityVsF()
