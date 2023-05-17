'''
@Author  ：Yan JP
@Created on Date：2023/5/5 22:51 
'''
import numpy as np
import para
from tiles import tile

def linear_normalization(x):
    # return (x - min(tiles.M_all_bound)) / (max(tiles.M_all_bound) - min(tiles.M_all_bound))
    return (x - 0) / (para.bitrate - 0)


def normalize(W):
    W = W/np.linalg.norm(W)
    return W


class transmission():
    def __init__(self,p):
        self.H_2=para.H_2
        self.p=p
        self.n0=para.N0
        self.B=para.B
        #--HMD
        self.F_max=para.F_max
        self.b_s=para.b_s
        self.Dt=0

        self.get_rt()

    def get_rt(self):
        sinr=self.p/self.n0*sum(self.H_2)
        self.rt=self.B*(np.log2(1+sinr))
        return self.rt
    def get_energy_t(self,Tu):
        return self.p*Tu

    def get_Tu(self,cil,Mil):
        num_bit=0
        for i in range(len(Mil)):
            num_bit+=(1-cil)*Mil+cil*Mil*para.co_ratio
        self.Tu=num_bit/self.rt

    def get_Td(self,cil,Mil):
        num_bit2=0
        for i in range(len(Mil)):
            num_bit2+=cil*Mil
        self.Td=num_bit2/(self.F_max*self.b_s)
        return self.Td

    def get_O(self,NF):  #生成NF个遮挡等级数
        # p=np.random.normal(0,2,NF)
        p = np.random.normal(0, 3, NF)
        p = np.abs(p)
        p = np.clip(p, 0, 5)
        self.O = np.floor(p).tolist()
        return self.O

    def get_Qi(self,Ni,sumN,Oi,dis_i):
        k=-(sumN/Ni-1)/5
        b=sumN/Ni
        w_oi=k*Oi+b
        zi=w_oi*Ni/sumN

        return zi*(1/dis_i)

    def get_Bt(self,):
        self.Bt=para.f/para.fps-self.Td-self.Tu
        return self.Bt

    def get_QoE(self,Q,Bt):
        self.QoE=para.a1*Q+para.a2*Bt
        return self.QoE

    def update_Dt(self,choose_Mil):
        self.Dt+=choose_Mil
        return self.Dt




class subEnvironment:
    def __init__(self, seed,ttile:tile,fov_id):
        self.fov_id = fov_id
        self.ttile = ttile
        self.fov = ttile.Fovs_tile_id[fov_id]
        self.fov_tile_id = ttile.Fovs_tile_id[fov_id]
        self.dis = ttile.dis[self.fov_tile_id]
        self.z = ttile.get_z(fov_id, self.fov_tile_id)
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

    def step(self,action,index):
        if action<5:
            k=1      #1表示压缩
        else: k=0
        l=action  #范围就是0-9

        Mil=self.action_value[l]*self.Mi

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
        q = 40 / dis_i * zi * li * (1 / Oi)
        reward = p + q
        self.tile_QoE.append(q)
        # reward = min(10, reward)
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
            r = 0.5 * li

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
