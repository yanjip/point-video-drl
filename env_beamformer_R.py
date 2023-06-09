'''
@Author  ：Yan JP
@Created on Date：2023/6/8 14:19 
'''
import numpy as np
import para
from tiles import tile


def linear_normalization(x):
    # return (x - min(tiles.M_all_bound)) / (max(tiles.M_all_bound) - min(tiles.M_all_bound))
    return (x - 0) / (para.bitrate - 0)


def normalize(W):
    W = W / np.linalg.norm(W, axis=0)  # 列归一化，不用reshape
    return W


z = 0.5


def fx(x):
    return np.exp(-z * x)


class upperEnvironmentBeam():
    def __init__(self):
        self.state_dim = para.K
        self.action_dim = para.K * para.N_aps
        self.times = 1000
        self.actionLow = 0
        # self.actionHigh = np.sqrt(para.maxPower)
        self.actionHigh = 1

        self.G = np.random.exponential(scale=1.0, size=(para.K, para.N_aps)).astype('float32')
        # self.G = self.G[:, np.argsort(self.G.sum(axis=0))]

        W = np.random.uniform(high=self.actionHigh, size=(para.K, para.N_aps)).astype('float32')
        # W = normalize(W)
        self.W = self.normalizeW(W)
        # 测试固定
        self.fix_W = self.W

        pass

    def normalizedActions(self, action):
        action = self.actionLow + (action + 1) * 0.5 * (self.actionHigh - self.actionLow)
        action = np.clip(action, self.actionLow, self.actionHigh)
        return action

    def reverse_action(self, action):
        action = 2 * (action - self.actionLow) / (self.actionHigh - self.actionLow) - 1
        action = np.clip(action, self.actionLow, self.actionHigh)
        return action

    def normalizeW(self, W):
        norm = np.linalg.norm(W, axis=0)
        if 0.0 in norm:
            norm = norm + 1e-4
        W = W / norm
        return W

    def sinr(self, ):
        W2 = np.square(self.W)
        gamma = np.zeros(para.K, dtype='float32')
        for k in range(para.K):
            numerator = np.dot(self.G[k, :].reshape(1, para.N_aps), self.W[k, :].reshape(para.N_aps, 1))
            numerator = np.square(np.linalg.norm(numerator))

            interference = 0
            for k1 in range(para.K):
                if k != k1:
                    interference += np.dot(self.G[k1, :].reshape(1, para.N_aps), self.W[k, :].reshape(para.N_aps, 1))
                    interference = np.square(np.linalg.norm(interference))
            denom = interference
            if denom == 0:
                denom += 1e-8
            gamma[k] = numerator / denom * np.sqrt(para.maxPower)
        return gamma.astype(np.float32)

    def reset(self):
        W = np.random.uniform(high=1, size=(para.K, para.N_aps)).astype('float32')
        self.W = self.normalizeW(W)
        # -----下面是另一种方法
        self.W = self.fix_W
        return self.sinr()

    def step(self, action_t):
        self.W = action_t.reshape(para.K, para.N_aps)
        self.W = self.normalizeW(self.W)

        next_state = self.sinr()
        reward = np.sum(np.log2(1 + next_state))
        reward -= para.K
        punish = self.check_power() / 2
        # if punish > 0:
        #     reward -= 5
        # -------解决UE3太小的问题
        # p_var=np.var(next_state)
        # reward-=p_var

        done = 0.0
        return next_state, reward, np.float32(done), None

    def check_power(self):
        sum_of_columns = np.sum(np.square(self.W), axis=0)
        p = 0
        for i in sum_of_columns:
            # if i - para.maxPower>0:
            #     print("hello")
            p += max(0, i - para.maxPower)
        return p
        pass

    def get_final_res(self, ):
        sinr = self.sinr()
        sum_sinr = np.sum(np.log2(1 + sinr))
        return sum_sinr

    def baseline_random(self, ):
        ddpg_sinr = self.get_final_res()
        self.base1 = np.random.uniform(high=1, size=(para.K, para.N_aps)).astype('float32')
        norm = np.linalg.norm(self.base1, axis=0)
        if 0.0 in norm:
            norm = norm + 1e-4
        self.base1 = self.base1 / norm
        self.W = self.base1
        base_sinr = self.get_final_res()
        return ddpg_sinr, base_sinr


class transmission():
    def __init__(self, p):
        self.H_2 = para.H_2
        self.p = p
        self.n0 = para.N0
        self.B = para.B
        # --HMD
        self.F_max = para.F_max
        self.b_s = para.b_s
        self.Dt = 0

        self.get_rt()

    def get_rt(self):
        sinr = self.p / self.n0 * sum(self.H_2)
        self.rt = self.B * (np.log2(1 + sinr))
        return self.rt

    def get_energy_t(self, Tu):
        return self.p * Tu

    def get_Tu(self, cil, Mil):
        num_bit = 0
        for i in range(len(Mil)):
            num_bit += (1 - cil) * Mil + cil * Mil * para.co_ratio
        self.Tu = num_bit / self.rt

    def get_Td(self, cil, Mil):
        num_bit2 = 0
        for i in range(len(Mil)):
            num_bit2 += cil * Mil
        self.Td = num_bit2 / (self.F_max * self.b_s)
        return self.Td

    def get_O(self, NF):  # 生成NF个遮挡等级数
        # p=np.random.normal(0,2,NF)
        p = np.random.normal(0, 3, NF)
        p = np.abs(p)
        p = np.clip(p, 0, 5)
        self.O = np.floor(p).tolist()
        return self.O

    def get_Qi(self, Ni, sumN, Oi, dis_i):
        k = -(sumN / Ni - 1) / 5
        b = sumN / Ni
        w_oi = k * Oi + b
        zi = w_oi * Ni / sumN

        return zi * (1 / dis_i)

    def get_Bt(self, ):
        self.Bt = para.f / para.fps - self.Td - self.Tu
        return self.Bt

    def get_QoE(self, Q, Bt):
        self.QoE = para.a1 * Q + para.a2 * Bt
        return self.QoE

    def update_Dt(self, choose_Mil):
        self.Dt += choose_Mil
        return self.Dt


class upperEnvironment():
    def __init__(self, ttile: tile, fov_id):
        self.state_dim = para.K * 2
        self.action_dim = para.K
        self.times = 30

    def reset(self):
        self.done = False
        self.obs_Bt = np.array([para.Bt] * para.K)
        self.obs_Q = np.array([0.0] * para.K)
        obs = np.concatenate((self.obs_Bt, self.obs_Q), axis=0)
        return obs

    def step(self, action, index):
        # self.obs_Bt-=(para.T_slot)
        self.obs_Bt -= (0.1)

        self.obs_Bt[self.obs_Bt < 0] = 0  # 将数组中小于0的元素赋值为0

        self.obs_Bt[action] += (para.f / para.fps)
        # 这里用下层策略 假设get_Q
        Q = np.random.randint(120, 150) / 100

        self.obs_Q[action] += Q
        # reward=2 * sum(self.obs_Q)+ 1 * sum(self.obs_Bt)
        reward = 0
        n_zeros = np.count_nonzero(self.obs_Bt == 0)  # 统计数组中等于0的元素的个数
        penaty = n_zeros * 2

        # 计算Q的方差，不宜过大
        # dis=max(self.obs_Q)-min(self.obs_Q)
        dis = np.var(self.obs_Q) + np.var(self.obs_Bt) * 2
        penaty += dis

        reward -= penaty

        if index >= self.times:
            self.done = True
        self.next_obs = np.concatenate((self.obs_Bt, self.obs_Q), axis=0)

        return self.next_obs, reward, self.done, None

    def get_info(self, ):
        return self.next_obs


class subEnvironment:
    def __init__(self, seed, ttile: tile, fov_id):
        self.fov_id = fov_id
        self.ttile = ttile
        self.fov = ttile.Fovs_tile_id[fov_id]
        self.fov_tile_id = ttile.Fovs_tile_id[fov_id]
        self.dis = ttile.dis[self.fov_tile_id]
        self.z = ttile.get_z(fov_id, self.fov_tile_id)
        self.zmin = min(self.z)
        self.zmax = max(self.z)
        # self.Q_fov=ttile.Q[self.fov_tile_id]    #暂时没有排序的必要

        self.Mi = para.bitrate

        # self.index=0

        self.trans = transmission(para.Pmax)

        self.action_value = [1.0, 0.8, 0.6, 0.4, 0.2, 0.2, 0.4, 0.6, 0.8, 1.0]  # 压缩->未压缩
        self.state_dim = 5  # Dt、-Tu、Td、Qi、
        self.action_dim = 10
        # self.observation_shape = (self.state_dim,)
        self.action_space = (self.action_dim,)
        # np.random.seed(seed)

        # self.Q_tile=Q_tile
        # shared buffer
        # self.max_buffer_size = max_buffer_size
        # self.batch_size = batch_size
        self.Dt = 0  # HMD的解码比特数
        self.Dt_nor = 0.0
        self.all_data = 0
        self.data_tiles_nor = 0.0
        self.reward = 0.0
        self.Dmax = para.F_max * para.b_s * para.T_slot

        self.get_order()
        self.get_Q()
        pass

    def get_z_nor(self, zi):
        return (zi - 0) / (self.zmax - self.zmin)

    def get_Q(self, ):
        Q = []
        for i in range(para.N_F):
            q = self.z[i] * (1 / self.ttile.dis[self.searchId[i]]) * self.ttile.O[self.searchId[i]]
            Q.append(q)
        self.Q = np.array(Q)

    def get_order(self, ):
        ##self.Q=np.sort(self.Q_fov)[::-1]
        # self.searchId=np.argsort(self.Q_fov)[::-1]

        # 默认排序：
        self.searchId = self.fov_tile_id
        pass

    def get_Index(self, index):
        return self.searchId[index]

    def reset(self):
        # 统计结果信息：
        self.tile_QoE = []
        self.tile_data = []
        self.Tu_Td = []

        self.Dt = 0
        self.Bt = 1  # 视频缓冲区 初始化为2s
        self.time_occu = 0.0
        self.done = 0
        observation = np.array([0.0, 0.0, 0.0, 0.0, 0])  ##Dt、Tu、Td、dis、zoi、l
        self.reward = 0.0
        self.all_data = 0
        self.all_data_nor = linear_normalization(self.all_data)
        # 计算Qi
        # self.index=0
        dis_i = self.ttile.dis[self.searchId[0]]
        zi = self.z[0]
        Oi = self.ttile.O[self.searchId[0]]

        obs = np.array([self.time_occu, self.Dt_nor, dis_i, zi, Oi])

        return obs

        pass

    def step(self, action, index):
        if action < 5:
            k = 1  # 1表示压缩
        else:
            k = 0
        l = action  # 范围就是0-9

        Mil = self.action_value[l] * self.Mi

        # 既然Tu算不了，就先把传输的比特数作为环境，归一化(可以算了
        data_tiles = k * Mil * para.co_ratio + (1 - k) * Mil
        self.tile_data.append(data_tiles)
        self.all_data += data_tiles
        self.all_data_nor = linear_normalization(self.all_data)
        Tu = data_tiles / self.trans.rt

        Mil_com = k * Mil * para.co_ratio
        self.Dt += Mil_com  #
        self.Dt_nor = linear_normalization(Mil_com)
        # 计算Td
        Td = Mil_com / (para.F_max * para.b_s)

        # 计算Qi
        dis_i = self.ttile.dis[self.searchId[index - 1]]
        zi = self.z[index - 1]
        Oi = self.ttile.O[self.searchId[index - 1]]

        li = abs(l - 4.5) + 0.5
        #########奖励设置################
        self.Bt = self.Bt + para.f / para.fps - para.T_slot
        self.time_occu += (Tu + Td) / para.T_slot

        penalty_t = (Tu + Td - para.T_slot)
        self.Tu_Td.append([Tu, Td])
        # dismax=1 zimax=1 limax=4.5
        p = 0
        # if penalty_t > 0:
        #     p = -10
        if self.time_occu > 1:
            p -= 25
        # q = 40 / dis_i * zi * li * (1 / Oi)
        zi_nor = self.get_z_nor(zi)
        q = fx(dis_i) * fx(Oi) * (zi_nor + li) * 10
        reward = p + q
        self.tile_QoE.append(q)
        reward = min(15, reward)
        # if k==0:
        #     rr=0.3*(li)
        #     reward+=rr
        #
        # reward*=10
        # self.Dmax=para.F_max*para.b_s*(para.T_slot-Tu)
        # if self.Dt>self.Dmax:
        # self.done=1
        # r=linear_normalization(self.Dt-self.Dmax) #后面再调(意思是若Dt越接近Dmax越好）
        r = 0
        Dt_Dmax_nor = linear_normalization(self.Dt - self.Dmax)
        if Dt_Dmax_nor < 0 and k == 1:
            # if self.Dt < self.Dmax and k == 1:
            # r=abs(reward)
            # r = li+4
            # r=0.1*(li+4)
            r = 0.2 * li

        # r =reward*li
        if k == 1 and self.Dt > self.Dmax:
            r = -5
        reward += r

        # self.index += 1

        if index == para.N_F:
            self.done = 1
            # self.index=0
        if self.done != 1:
            dis_i = self.ttile.dis[self.searchId[index]]
            zi = self.z[index]
            # Dt_nor=linear_normalization(self.Dt)
            # Dt_Dmax_nor = linear_normalization(self.Dt - self.Dmax)
            next_obs = np.array([self.time_occu, Dt_Dmax_nor, dis_i, zi, Oi])
        else:
            next_obs = np.array([0., 0., 0., 0., 0.])

        return next_obs, reward, self.done, None

    def get_info(self, ):
        # 设置打印选项，保留2位小数
        np.set_printoptions(precision=2)

        print("tile_id：", self.fov_tile_id)
        print("distance：", self.dis)
        if type(self.ttile.O) == list:
            O = np.array(self.ttile.O)
        else:
            O = self.ttile.O
        print("遮挡等级：", O[self.searchId])
        print("tile_datasize:", self.tile_data)
        tile_QoE = np.array(self.tile_QoE)
        print("Tu_Td:", self.Tu_Td)
        print("time_consum:", self.time_occu)
        print("QoE:", tile_QoE)
        print("sum_QoE:", sum(self.tile_QoE), end='\n\n')

        pass
