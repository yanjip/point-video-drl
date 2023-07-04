'''
@Author  ：Yan JP
@Created on Date：2023/5/5 22:52 
'''
import math
import numpy as np
import scipy.io as sio

# Data size scales
BYTE = 8
KB = 1024 * BYTE
MB = 1024 * KB
GB = 1024 * MB
# TB = 1024*GB
# PB = 1024*TB

KHZ = 1e3
MHZ = KHZ*1e3
GHZ = MHZ*1e3

N_aps = 4  # APs

# Channels
B = 40 * MHZ  # MHz
N0 = 1e-9  # 单位：W The variance of complex white Gaussian channel noise
# 生成信道数据
# H = np.random.rayleigh(scale=1, size= self.N)*1e-3
# H = np.random.rayleigh(scale=1, size= self.N)
H_2 = np.random.exponential(1e-6, N_aps)

#tile
N_all = np.array([100, 200]) * 1e3  # points
M_all_bound=np.array([10,20])*1024*1024
N=3*3*4   #all tiles
L=5       # level
N_fovs=9   #numbers of fovs
N_F=12      #each requested numbers of tiles
zipf = 1.1  # 齐普夫参数
bitrate = 30 * 1024 * 1024 * 8 / N  # bps
f = 30  # 帧数
fps=30
M=bitrate*(f / fps)

Pmax = 60  # BS的功率上限
co_ratio = 0.8  # compress_ratio  Tran-RL那篇文章也是用的0.8
# M=np.array([])
# f_fps=1.0
action_value = [0.2, 0.4, 0.6, 0.8, 1.0]
quality_level = {1: 0.2, 2: 0.4, 3: 0.6, 4: 0.8, 5: 1.0}

a1 = 0.5
a2 = 0.5

T_slot = 0.3

# UE
F_max = 0.08e9  # cycles/s MEC那篇论文为0.5G    0.3e8
b_s = 0.005e3  # bits/cycle  MEC   之前0.02
D_max = 1 * GB  # HMD的处理区数据量大小

K = 3  # user number
Bt = 2.0  # 设备的初始化视频缓冲区
QoE0 = 10
sinr = 44.0

# upper para
maxPower = 10.0
area_size = 500
max_timestamp = 30

# basline
n_power_levels = 5


def cul_r(H_s, p):
    sinr = sum(H_s) / N0 * p
    return np.log2(1 + sinr) * B


def px(x):
    return N0 * (math.pow(2, x / B) - 1)


# import pickle
# with open('tiles.pkl', 'rb') as inp:
#     ttile = pickle.load(inp)
# print(ttile)


def get_codebook():
    mdict = sio.loadmat('beamforming/codebook_dft44.mat')
    code_book = mdict['ans']
    # code_book = mdict['W']
    c1 = np.transpose(code_book[:, [0, 1, 2]])
    c2 = np.transpose(code_book[:, [0, 1, 3]])
    c3 = np.transpose(code_book[:, [0, 2, 3]])
    c4 = np.transpose(code_book[:, [1, 2, 3]])
    stacked_matrix = np.stack((c1, c2, c3, c4))
    return stacked_matrix

# get_codebook()
