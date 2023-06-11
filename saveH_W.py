'''
@Author  ：Yan JP
@Created on Date：2023/6/10 16:50 
'''
import numpy as np
import os
import pickle


def save(H, W, file_path):
    if os.path.exists(file_path):
        data = load(file_path)
    else:
        data = {'H': [], 'W': []}
    data['H'].append(H)
    data['W'].append(W)
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    pass


def load(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data


if __name__ == '__main__':
    arr1 = np.array([1, 2])
    arr2 = np.array([6, 7])
    save(arr1, arr2, 'H_W/arrays.npz')
    print(load('H_W/arrays.npz'))
