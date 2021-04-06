"""
基于神经网络的全局运动估计
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import device
import numpy as np
import pickle
import random
import time
import cv2
import matplotlib.pyplot as plt
from lib import flowlib as fl


class NetB(nn.Module):
    def __init__(self, in_dim, hid1_dim, hid2_dim, out_dim):
        super(NetB, self).__init__()
        self.hidden1 = nn.Linear(in_dim, hid1_dim)
        self.hidden2 = nn.Linear(hid1_dim, hid2_dim)
        self.pred = nn.Linear(hid2_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.pred(x)
        return x


class NetA(nn.Module):
    def __init__(self, in_dim, hid1_dim, out_dim):
        super(NetA, self).__init__()
        self.hidden1 = nn.Linear(in_dim, hid1_dim)
        self.pred = nn.Linear(hid1_dim, out_dim)

    def forward(self, x):
        x = F.relu(self.hidden1(x))
        x = self.pred(x)
        return x


def result_refine(res):
    res_sort = np.sort(res, axis=0)
    res_refine = np.mean(res_sort[22:42, :], axis=0).reshape(1, -1)
    res_refine = np.repeat(res_refine, 64, axis=0)
    return res_refine


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读取光流图参数 ground truth 文件
    with open('D:/TMM_exp/global_param_gtB_test.pkl', 'rb') as f:
        data_dict = pickle.load(f)
        path_list = list(data_dict.keys())
    epe = 0.0
    count = 0

    # 模型 backbone 选择 A/B
    net = NetA(64, 128, 64)
    # net = NetB(64, 128, 128, 64)

    # 训练好的模型路径
    net.load_state_dict(torch.load(r'D:/学习资料/PR2020/global_motion_estimation/model/netA_modelB.pkl'))
    net.eval()
    net.to(device)

    loss_list = []

    for i in range(len(path_list)):
        count += 1
        data_flow = np.load(path_list[i]).astype(np.float32)
        data_flow = cv2.resize(data_flow, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
        para = data_dict[path_list[i]][0]
        para_x = para[0]
        para_y = para[1]
        data_new = np.arange(0, 64)

        # x 分量运动场标签生成
        data_com_x = para_x[0] * data_new.reshape(1, -1) ** 2 + para_x[1] * data_new.reshape(1, -1) + para_x[2]
        # data_com_x = para_x[0] * data_new.reshape(1, -1) + para_x[1]
        label_x = np.repeat(data_com_x, 64, axis=0)

        # y 分量运动场标签生成
        data_com_y = para_y[0] * data_new.reshape(1, -1) ** 2 + para_y[1] * data_new.reshape(1, -1) + para_y[2]
        # data_com_y = para_y[0] * data_new.reshape(1, -1) + para_x[1]
        label_y = np.repeat(data_com_y, 64, axis=0)

        # 对x y 分量分别进行计算
        start1 = time.time()
        for channel in range(2):
            inputs = data_flow[:, :, channel]
            if channel == 0:
                inputs = inputs[:, np.newaxis, :]
            else:
                inputs = inputs.T[:, np.newaxis, :]

            inputs = torch.tensor(inputs).type(torch.FloatTensor)
            inputs = inputs.to(device)
            with torch.no_grad():

                outputs = net(inputs)

                if channel == 0:
                    x1 = outputs.cpu().numpy().squeeze()
                    x1 = result_refine(x1)
                else:
                    y1 = outputs.cpu().numpy().squeeze()
                    y1 = result_refine(y1)
        end1 = time.time()
        print('inference time{}'.format(end1 - start1))
        # 误差计算
        epe += fl.flow_error(x1, y1, label_x, label_y)
        # print(epe)
    print('average: {}'.format(epe/count))