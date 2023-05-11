'''
@Author  ：Yan JP
@Created on Date：2023/5/5 22:55 
'''
import para
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.nn.functional as F
from torch import optim
import random

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(para.H)