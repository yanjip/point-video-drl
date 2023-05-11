'''
@Author  ：Yan JP
@Created on Date：2023/5/5 22:51 
'''
import numpy as np
import para
from tiles import tile

def linear_normalization(x):
    # return (x - min(tiles.M_all_bound)) / (max(tiles.M_all_bound) - min(tiles.M_all_bound))
    return (x - 0) / (max(para.M_all_bound) - 0)


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
        self.fov_id=fov_id
        self.ttile=ttile
        self.fov =ttile.Fovs[fov_id]
        self.fov_tile_id=ttile.Fovs_tile_id[fov_id]
        self.dis=ttile.dis[self.fov_tile_id]
        self.z=ttile.z[self.fov_tile_id]
        self.Q_fov=ttile.Q[self.fov_tile_id]    #暂时没有排序的必要

        self.Mi=para.bitrate

        self.index=0

        self.trans=transmission(para.Pmax)

        self.action_value=[1.0,0.8,0.6,0.4,0.2, 0.2,0.4,0.6,0.8,1.0] #压缩->未压缩
        self.state_dim = 4   #Dt、-Tu、Td、Qi、
        self.action_dim = 10
        # self.observation_shape = (self.state_dim,)
        self.action_space = (self.action_dim,)
        np.random.seed(seed)

        # self.Q_tile=Q_tile
        # shared buffer
        # self.max_buffer_size = max_buffer_size
        # self.batch_size = batch_size
        self.Dt=0     #HMD的解码比特数
        self.Dt_nor=0.0
        self.all_data=0
        self.data_tiles_nor=0.0
        self.reward=0.0
        self.Dmax=para.F_max*para.b_s*para.T_slot

        self.get_order()

    def get_order(self,):
        ##self.Q=np.sort(self.Q_fov)[::-1]
        # self.searchId=np.argsort(self.Q_fov)[::-1]

        #默认排序：
        self.searchId=self.fov_tile_id
        pass
    def get_Index(self,):
        return self.searchId[self.index]

    def reset(self):
        self.done=0
        observation=np.array([0.0,0.0,0.0,0.0,0])   ##Dt、Tu、Td、dis、zoi、l
        self.reward=0.0
        self.all_data_nor = linear_normalization(self.all_data)
        #计算Qi
        self.index=0
        dis_i=self.ttile.dis[self.searchId[self.index]]
        zi=self.ttile.z[self.searchId[self.index]]

        obs=np.array([self.all_data_nor,self.Dt_nor,dis_i,zi])

        return obs

        pass

    def step(self,action):
        if action<5:
            k=1      #1表示压缩
        else: k=0
        l=action  #范围就是0-9

        Mil=self.action_value[l]*self.Mi

        # 既然Tu算不了，就先把传输的比特数作为环境，归一化(可以算了
        data_tiles = k * Mil * para.co_ratio + (1 - k) * Mil
        self.all_data += data_tiles
        self.all_data_nor = linear_normalization(self.all_data)
        Tu=data_tiles/self.trans.rt

        Mil_com=k*Mil*para.co_ratio
        self.Dt+=Mil_com       #
        self.Dt_nor=linear_normalization(Mil_com)
        #计算Td
        Td=Mil_com/(para.F_max*para.b_s)

        #计算Qi
        dis_i=self.ttile.dis[self.searchId[self.index]]
        zi=self.ttile.z[self.searchId[self.index]]

        li=abs(l-4.5)
        reward=-Tu -Td + dis_i*zi*li

        self.Dmax=para.F_max*para.b_s*(para.T_slot-Tu)
        if self.Dt>self.Dmax:
            # self.done=1
            r=linear_normalization(self.Dt-self.Dmax) #后面再调(意思是若Dt越接近Dmax越好）
            reward-=r

        self.index += 1

        if self.index == para.N_F :
            self.done=1
            self.index=0
        if self.done!=1:
            dis_i=self.ttile.dis[self.searchId[self.index]]
            zi=self.ttile.z[self.searchId[self.index]]
            Dt_nor=linear_normalization(self.Dt)
            next_obs=np.array([self.all_data_nor,Dt_nor,dis_i,zi])
        else:next_obs=None

        return next_obs,reward,self.done,None


        pass


