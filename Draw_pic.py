'''
@Author  ：Yan JP
@Created on Date：2023/6/5 17:20 
'''
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import random
import torch
import pandas as pd

from matplotlib.font_manager import FontProperties  # 导入字体模块


# 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
def chinese_font():
    try:
        font = FontProperties(
            # 系统字体路径
            fname='C:\\Windows\\Fonts\\方正粗黑宋简体.ttf', size=14)
    except:
        font = None
    return font


# 中文画图
def plot_rewards_cn(rewards, cfg, path=None, tag='train'):
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(cfg['env_name'],
                                       cfg['algo_name']), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(smooth(rewards))
    plt.legend(('奖励', '滑动平均奖励',), loc="best", prop=chinese_font())
    if cfg['save_fig']:
        plt.savefig(f"{path}/{tag}ing_curve_cn.png")
    if cfg['show_fig']:
        plt.show()


# 用于平滑曲线，类似于Tensorboard中的smooth
def smooth(data, weight=0.9):
    '''
    Args:
        data (List):输入数据
        weight (Float): 平滑权重，处于0-1之间，数值越高说明越平滑，一般取0.9

    Returns:
        smoothed (List): 平滑后的数据
    '''
    last = data[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, cfg, path=None, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title(f"{tag}ing curve on {cfg['device']} ")
    plt.title("Beamforming Solved By DDPG")
    plt.xlabel('Epsiodes')
    plt.ylabel('Reward')
    plt.plot(rewards, label='rewards')
    plt.plot(smooth(rewards), label='smoothed')
    plt.legend()
    if cfg['save_fig']:
        plt.savefig(f"{path}/{tag}ing_curve.png")
    if cfg['show_fig']:
        plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


# 保存奖励
def save_results(res_dic, tag='train', path=None):
    '''
    '''
    Path(path).mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(res_dic)
    df.to_csv(f"{path}/{tag}ing_results.csv", index=None)
    print('结果已保存: ' + f"{path}/{tag}ing_results.csv")


# 创建文件夹
def make_dir(*paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


# 删除目录下所有空文件夹
def del_empty_dir(*paths):
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# 保存参数
def save_args(args, path=None):
    Path(path).mkdir(parents=True, exist_ok=True)
    with open(f"{path}/params.json", 'w') as fp:
        json.dump(args, fp, cls=NpEncoder)
    print("参数已保存: " + f"{path}/params.json")


import time


def write_sinr(sinr, SE):
    with open('H_W/sinr_SE.txt', 'a+') as F:
        F.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())) + '\n')
        F.write("SINR:" + str(sinr) + "       SE_all:" + str(SE) + "\n\n")


if __name__ == '__main__':
    sinr = [11.379501, 4.522286, 41.942398]
    se = 11.519477844238281
    write_sinr(sinr, se)
