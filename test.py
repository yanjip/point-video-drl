'''
@Author  ：Yan JP
@Created on Date：2023/5/6 20:45 
'''
import para
import numpy as np

# M_all = np.random.normal(15*para.MB, 10000000, size=10)
# print(M_all)

# p = np.random.normal(0, 3, 12)
# p=np.abs(p)
# p=np.clip(p,0,5)
# # print(p)
# O=np.floor(p).tolist()
# print(type(O))

# np.save('runs/DQN/reward.npy', np.array([1, 1, 2]))

# def f1():
#     np.random.seed(123)
#     print(np.random.randint(10))
#     print(np.random.randint(5))
#
#
# def f2():
#     print(np.random.randint(11))
#
#
# f1()
# f2()
# f2()

a = np.array([3, 4, 5])
b = a > 3
print(type(b))

import datetime

a = datetime.datetime.now().strftime("%Y/%m/%d-%H:%M:%S")
print(a)

'''待用
self.powerbook = np.arange(Config().n_power_levels) * self.max_power / (Config().n_power_levels - 1)
self.power_index = action % Config().n_power_levels
self.code_index = action // Config().n_power_levels
self.code = self.codebook[:, self.code_index]
self.power = self.powerbook[self.power_index]
'''
