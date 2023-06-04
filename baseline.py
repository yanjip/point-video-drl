'''
@Author  ：Yan JP
@Created on Date：2023/5/17 16:39 
'''
import numpy as np
import para
from tiles import tile
from env import transmission


def linear_normalization(x):
    # return (x - min(tiles.M_all_bound)) / (max(tiles.M_all_bound) - min(tiles.M_all_bound))
    return (x - 0) / (para.bitrate - 0)


def normalize(W):
    W = W / np.linalg.norm(W)
    return W


z = 0.5


def fx(x):
    return np.exp(-z * x)


class greedyMethod():
    def __init__(self, ttile: tile, fov_id):
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

        self.Dt = 0  # HMD的解码比特数
        self.all_data = 0
        self.data_tiles_nor = 0.0
        # self.Dmax = para.F_max * para.b_s * para.T_slot/10
        self.Dmax = 0
        self.searchId = self.fov_tile_id

        self.actions = [5] * para.N_F

        self.QoE = 0

        # self.get_Q()

        pass

    def reset(self):
        # 统计结果信息：
        self.tile_QoE = []
        self.tile_data = []
        self.Tu_Td = []

        self.Dt = 0
        self.Bt = 1  # 视频缓冲区 初始化为2s
        self.time_occu = 0.0
        self.done = 0
        self.reward = 0.0
        self.all_data = 0
        self.all_data_nor = linear_normalization(self.all_data)

        data_tiles = self.action_value[5] * self.Mi * para.N_F
        self.Tu = data_tiles / self.trans.rt
        self.Td = 0.0
        self.time_occu += (self.Tu + self.Td) / para.T_slot

    def get_QoE(self, dis_i, Oi, zi_nor, li):
        q = fx(dis_i) * fx(Oi) * (zi_nor + li) * 10
        self.QoE += q
        pass

    def step(self, index):
        # a = self.actions[index]
        dis_i = self.ttile.dis[self.searchId[index]]
        zi = self.z[index]
        Oi = self.ttile.O[self.searchId[index]]
        zi_nor = self.get_z_nor(zi)
        print(self.time_occu)
        if self.Dmax > self.Dt and self.time_occu < 1:
            self.actions[index] = 0
            Mil_com = self.action_value[0] * self.Mi * para.co_ratio
            self.Dt += Mil_com
            Td = (Mil_com / (para.F_max * para.b_s)) / para.T_slot
            Tu = (Mil_com - self.action_value[5] * self.Mi) / self.trans.rt
            self.Tu_Td.append([Td, Tu])
            self.time_occu += ((Tu + Td) / para.T_slot)
            li = abs(0 - 4.5) + 0.5
            self.get_QoE(dis_i, Oi, zi_nor, li)

            return
        if self.time_occu < 1:
            self.actions[index] = 9
            Mil = self.action_value[9] * self.Mi
            Tu = (Mil - self.action_value[5] * self.Mi) / self.trans.rt
            self.time_occu += (Tu / para.T_slot)
            li = abs(9 - 4.5) + 0.5
            self.get_QoE(dis_i, Oi, zi_nor, li)
            self.Tu_Td.append([Tu, 0])

            return
        li = 1
        # self.Tu_Td.append([Tu, 0])
        self.get_QoE(dis_i, Oi, zi_nor, li)

        pass

    def get_info(self, ):
        print("action:", self.actions)
        print("QoE:", self.QoE)
        print("time_consum:", self.time_occu)

        pass

    def get_z_nor(self, zi):
        return (zi - 0) / (self.zmax - self.zmin)