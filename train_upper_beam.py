'''
@Author  ：Yan JP
@Created on Date：2023/6/4 10:59
'''
from DDPG import *
import os
import env
import para
from torch.utils.tensorboard import SummaryWriter
from Draw_pic import *
import saveH_W

def create_env_agent(arg_dict):
    env_beam = env.upperEnvironmentBeam()  # 装饰action噪声
    n_states = env_beam.state_dim
    n_actions = env_beam.action_dim
    agent = DDPG(n_states, n_actions, arg_dict)
    return env_beam, agent


def train(arg_dict, env_beam, agent):
    # 开始计时
    startTime = time.time()
    print("开始训练智能体......")
    ou_noise = OUNoise(env_beam.action_dim, decay_period=arg_dict["train_eps"])  # noise of action
    ou_noise.reset()
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'runs/DQN_upper_beam/' + current_time
    # writer = SummaryWriter(log_dir=train_log_dir)

    for i_ep in range(arg_dict['train_eps']):
        # agent.var =max(0.2,agent.var* 0.995)
        state = env_beam.reset()
        ou_noise.reset()
        done = False
        ep_reward = 0
        i_step = 0
        timestamp = 0
        while not done:
            i_step += 1
            timestamp += 1
            action = agent.choose_action(state)
            action = ou_noise.get_action(action, i_ep)
            # action = 0 + (ou_noise.high - 0) * (action + 1) / 2  # 将动作映射到（0，high）
            # action= np.clip(np.random.normal(action, agent.var), 0, ou_noise.high)
            next_state, reward, done, _ = env_beam.step(action)
            # if timestamp >= para.max_timestamp and env_beam.check_power() == 0:
            if timestamp >= para.max_timestamp:
                done = np.float32(1.0)
                # reward *= 10
            ep_reward += reward
            agent.memory.push(state, action, reward, next_state, done)
            agent.update()
            state = next_state
        # writer.add_scalar('step_rewards:', ep_reward, global_step=i_ep)

        if (i_ep + 1) % 2 == 0:
            # print("W:",env_beam.W)
            print(
                f'Env_beam:{i_ep + 1}/{arg_dict["train_eps"]}, Reward:{ep_reward :.2f},SINR:{env_beam.best_sinr},sigma:{ou_noise.sigma}')
            # print(f'Env_beam:{i_ep + 1}/{arg_dict["train_eps"]}, Reward:{ep_reward :.2f}')
            d, b = env_beam.baseline_random()
            print(f'DDPG:{d},*******,baseline:{b}')
            print("--------------------------------\n")

        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print("W:", env_beam.W)
    # saveH_W.save_res(env_beam.final_res)

    print('训练结束 , 用时: ' + str(time.time() - startTime) + " s")
    # 关闭环境
    return {'episodes': range(len(rewards)), 'rewards': rewards, 'ma_rewards': ma_rewards}


# 测试函数
def test(arg_dict, env_beam, agent):
    startTime = time.time()
    print("开始测试智能体......")
    # print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(arg_dict['test_eps']):
        env_beam = env.upperEnvironmentBeam()  # 装饰action噪声
        state = env_beam.reset()
        done = False
        ep_reward = 0
        i_step = 0
        timestamp = 0
        while not done:
            i_step += 1
            timestamp += 1
            action = agent.choose_action(state)
            # action = ou_noise.get_action(action, i_step)
            # action = 0 + (np.sqrt(para.maxPower) - 0) * (action + 1) / 2  # 将动作映射到（0，high）
            next_state, reward, done, _ = env_beam.step(action)
            if timestamp >= para.max_timestamp:
                print("W:", env_beam.W, end='\n\n')
                done = np.float32(1.0)
            ep_reward += reward
            state = next_state
        # if (i_ep + 1) % 2 == 0:
        #     print(f'Env_beam:{i_ep + 1}/{arg_dict["test_eps"]}, Reward:{ep_reward:.2f}')
        # rewards.append(ep_reward)
        # if ma_rewards:
        #     ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        # else:
        #     ma_rewards.append(ep_reward)
        print(f"Epside:{i_ep + 1}/{arg_dict['test_eps']}, Reward:{ep_reward:.1f},SINR:{next_state[-para.K:]}")
        d, b = env_beam.baseline_random()
        print(f'DDPG:{d},*******,baseline:{b}')
        rewards.append(d)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * d)
        else:
            ma_rewards.append(d)
        print("*" * 60)

    print("测试结束 , 用时: " + str(time.time() - startTime) + " s")
    return {'episodes': range(len(rewards)), 'rewards': rewards}


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    # 相关参数设置
    parser = argparse.ArgumentParser(description="hyper parameters")
    parser.add_argument('--algo_name', default='DDPG', type=str, help="name of algorithm")
    parser.add_argument('--train_eps', default=150, type=int, help="episodes of training")  # 原本150
    parser.add_argument('--test_eps', default=70, type=int, help="episodes of testing")
    parser.add_argument('--gamma', default=0.99, type=float, help="discounted factor")
    parser.add_argument('--critic_lr', default=1e-3, type=float, help="learning rate of critic")
    parser.add_argument('--actor_lr', default=1e-4, type=float, help="learning rate of actor")
    parser.add_argument('--memory_capacity', default=2000, type=int, help="memory capacity")  # 原本8000  500
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--target_update', default=2, type=int)
    parser.add_argument('--soft_tau', default=1e-2, type=float)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--var', default=5.0, type=float)
    parser.add_argument('--device', default='cuda', type=str, help="cpu or cuda")
    parser.add_argument('--seed', default=520, type=int, help="seed")
    parser.add_argument('--show_fig', default=False, type=bool, help="if show figure or not")
    parser.add_argument('--save_fig', default=True, type=bool, help="if save figure or not")
    args = parser.parse_args()

    # 将参数转化为字典 type(dict)
    arg_dict = {**vars(args)}
    print("算法参数字典:", arg_dict)

    train_flag = True
    test_flag = False
    # train_flag = False
    # test_flag = True
    # -------------------------训练----------------------------------------------------#
    if train_flag:
        # 创建环境和智能体
        env_beam, agent = create_env_agent(arg_dict)
        # 传入算法参数、环境、智能体，然后开始训练
        res_dic = train(arg_dict, env_beam, agent)
        print("算法返回结果字典:", res_dic)
        # 保存相关信息

        agent.save_model(path='runs/model/')
        # save_args(arg_dict, path=arg_dict['result_path'])
        # save_results(res_dic, tag='train', path='runs/DQN_upper_beam')
        plot_rewards(res_dic['rewards'], arg_dict, path='runs/DQN_upper_beam', tag="train")

    # ---------------------------------测试---------------------------------------------------#
    if test_flag:
        # # =================================================================================================
        # 创建新环境和智能体用来测试
        print("=" * 300)

        env_beam, agent = create_env_agent(arg_dict)
        # 加载已保存的智能体
        agent.load_model(path='runs/model/upper_agent_UE3.pt')
        res_dic = test(arg_dict, env_beam, agent)

        # save_results(res_dic, tag='test', path='runs/DQN_upper_beam')
        plot_rewards(res_dic['rewards'], arg_dict, path='runs/DQN_upper_beam', tag="test")
