'''
@Author  ：Yan JP
@Created on Date：2023/5/11 15:36 
'''


def step(self, action):
    k = action[0]  # 0表示未压缩
    l = action[1]

    Mil = self.action_value[l] * self.Mi

    # 既然Tu算不了，就先把传输的比特数作为环境，归一化(可以算了
    data_tiles = k * Mil * para.co_ratio + (1 - k) * Mil
    self.all_data += data_tiles
    self.all_data_nor = linear_normalization(self.all_data)
    Tu = data_tiles / self.trans.rt

    Mil_com = k * Mil * para.co_ratio
    self.Dt += Mil_com  #
    self.Dt_nor = linear_normalization(Mil_com)
    # 计算Td
    Td = Mil_com / (para.F_max * para.b_s)

    # 计算Qi
    dis_i = self.ttile.dis[self.index]
    zi = self.ttile.z[self.index]

    reward = -Tu - Td + dis_i * zi * l
    self.index += 1

    self.Dmax = para.F_max * para.b_s * (para.T_slot - Tu)
    if self.Dt > self.Dmax:
        # self.done=1
        r = linear_normalization(self.Dt - self.Dmax) * 10  # 后面再调(意思是若Dt越接近Dmax越好）
        reward -= r

    if self.index == para.N_F:
        self.done = 1

    dis_i = self.ttile.dis[self.index]
    zi = self.ttile.z[self.index]
    Dt_nor = linear_normalization(self.Dt)
    next_obs = np.array([self.all_data_nor, Dt_nor, dis_i, zi])

    return next_obs, reward, self.done, None
