'''
@Author  ：Yan JP
@Created on Date：2023/6/4 14:53 
'''
import numpy as np
import math
import para


class Channels:
    def __init__(self):
        self.distance = np.random.randint(1, para.area_size, size=(para.K, para.N_aps))
