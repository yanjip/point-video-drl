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
import pickle


def create_env_agent(arg_dict):
    env_beam = env.upperEnvironmentBeam()  # 装饰action噪声
    n_states = env_beam.state_dim
    n_actions = env_beam.action_dim
    agent = DDPG(n_states, n_actions, arg_dict)
    return env_beam, agent


def train(arg_dict, env_beam, agent):
    # 开始计时
    seed = 10
    np.random.seed(seed)
    torch.manual_seed(seed)
    startTime = time.time()
    print("开始训练智能体......")
    ou_noise = OUNoise(env_beam.action_dim, decay_period=arg_dict["train_eps"])  # noise of action
    ou_noise.reset()
    rewards = []  # 记录所有回合的奖励
    final_SE = 0
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = 'runs/DDPG_upper_beam/' + current_time
    # writer = SummaryWriter(log_dir=train_log_dir)
    totol_step = arg_dict['train_eps']
    randomBase = []
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
        d, b = env_beam.baseline_random()
        if d > final_SE and env_beam.second_largest > 10:
            final_SE = d
            final_W = env_beam.W
            final_sinr = env_beam.best_sinr

        if (i_ep + 1) % 2 == 0:
            # print("W:",env_beam.W)
            print(
                f'Env_beam:{i_ep + 1}/{arg_dict["train_eps"]}, Reward:{ep_reward :.2f},SINR:{env_beam.best_sinr},sigma:{ou_noise.sigma}')
            # print(f'Env_beam:{i_ep + 1}/{arg_dict["train_eps"]}, Reward:{ep_reward :.2f}')
            print(f'DDPG:{d},*******,baseline:{b}')
            print("--------------------------------\n")
        if totol_step - i_ep < 20:
            randomBase.append(b)
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
    print(f'Final_SINR:{final_sinr},Final_SE:{final_SE}')
    print("Final_W:", final_W)
    # write_sinr(env_beam.best_sinr, d)
    write_sinr(final_sinr, final_SE, sum(randomBase) / 20)

    # saveH_W.save_res(env_beam.final_res)

    print('训练结束 , 用时: ' + str(time.time() - startTime) + " s")
    # 关闭环境
    return {'episodes': range(len(rewards)), 'rewards': rewards, 'ma_rewards': ma_rewards}


# 测试函数
def test(arg_dict, env_beam, agent, test_eps):
    startTime = time.time()
    print("开始测试智能体......")
    # print(f"环境名: {arg_dict['env_name']}, 算法名: {arg_dict['algo_name']}, Device: {arg_dict['device']}")
    rewards = []  # 记录所有回合的奖励
    ma_rewards = []  # 记录所有回合的滑动平均奖励
    for i_ep in range(test_eps):
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


def testBeamDqn():
    env_beam_DQN = env.BeamformBL(env_beam)
    state = env_beam_DQN.reset()
    done = False
    episode_reward = 0
    res = []
    while not done:
        action = agent2.choose_action(state, epsilon=0)
        res.append(action)
        next_state, reward, done, _ = env_beam_DQN.step(action)
        state = next_state
        episode_reward += reward
    print("-----------------evaluate_reward:{} \t \n SINR: {} \t SE:{}\n".format(
        episode_reward, next_state, reward * 10))
    return next_state, reward * 10


if __name__ == '__main__':
    # 防止报错 OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # 获取当前路径
    curr_path = os.path.dirname(os.path.abspath(__file__))
    # 获取当前时间
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d-%H_%M_%S")

    train_flag = False
    test_flag = True

    # ---------------------------------测试---------------------------------------------------#
    if test_flag:
        # # =================================================================================================
        # 创建新环境和智能体用来测试
        print("=" * 300)
        env_beam = env.upperEnvironmentBeam()
        # 加载已保存的智能体
        # agent.load_model(path='runs/model/upper_agent_UE3.pt')
        # test_eps=2
        # res_dic = test(arg_dict, env_beam, agent,test_eps)

        # save_results(res_dic, tag='test', path='runs/DDPG_upper_beam')
        # plot_rewards(res_dic['rewards'], arg_dict, path='runs/DDPG_upper_beam', tag="test")
        #
        # env_beam, agent = create_env_agent(arg_dict)

        # ---------------baseline---------------
        with open('runs/model/agent_beam_DQN_7_13.pkl', 'rb') as f:
            agent2 = pickle.load(f)
        power = [37, 39, 41, 42,
                 43]  # 5.011872336272722  7.943282347242816  12.589254117941675  15.848931924611133  19.952623149688797

        for p in power:
            para.maxPower = p
            sinr, SE = testBeamDqn()
