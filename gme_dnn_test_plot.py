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
import cv2
import matplotlib.pyplot as plt
from lib import flowlib as fl
from lib import flowlib_v2 as fl2


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
    res_refine = np.mean(res_sort[20:45, :], axis=0).reshape(1, -1)
    res_refine = np.repeat(res_refine, 64, axis=0)
    return res_refine


def vis_RGB(flow, min_v, max_v):
    factor = 255 / (max_v - min_v)
    res = (flow - min_v) * factor
    return res.astype(np.uint8)


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 读取光流图参数 ground truth 文件
    # with open('D:/TMM_exp/global_param_gtA_test.pkl', 'rb') as f:
    #     data_dict = pickle.load(f)
    #     path_list = list(data_dict.keys())
    epe = 0.0
    count = 0

    # 模型 backbone 选择 A/B
    net = NetA(64, 128, 64)
    # net = NetB(64, 128, 128, 64)

    # 训练好的模型路径
    net.load_state_dict(torch.load(r'D:/code/yz_job/code/model/netA_modelB.pkl'))
    net.eval()
    net.to(device)

    loss_list = []
    path = r'D:\code\yz_job\code\optical_flow\6.npy'

    data_flow = np.load(path).astype(np.float32)
    u = data_flow[:, :, 0]
    v = data_flow[:, :, 1]
    rad = np.sqrt(u ** 2 + v ** 2)  # 光流场方向
    maxrad = max(-1, np.max(rad))
    fl.visualize_flow(data_flow)
    data_flow = cv2.resize(data_flow, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)

    flow_x = data_flow[:, :, 0].squeeze()
    flow_y = data_flow[:, :, 1].squeeze()

    flow_x_vis = cv2.resize(vis_RGB(flow_x, np.min(flow_x), np.max(flow_x)), dsize=(300, 200), interpolation=cv2.INTER_LINEAR)
    flow_y_vis = cv2.resize(vis_RGB(flow_y, np.min(flow_y), np.max(flow_y)), dsize=(300, 200),
                            interpolation=cv2.INTER_LINEAR)

    # cv2.imshow("name", flow_x_vis)
    # cv2.waitKey(0)
    #
    # cv2.imshow("name", flow_y_vis)
    # cv2.waitKey(0)

    # 对x y 分量分别进行计算
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
                pred_x = outputs.cpu().numpy().squeeze()
                pred_x_refine = result_refine(pred_x)
            else:
                pred_y = outputs.cpu().numpy().squeeze()
                pred_y_refine = result_refine(pred_y)

    pred_x_color = pred_x[:, :, np.newaxis]
    pred_y_color = pred_y.T[:, :, np.newaxis]
    flo_pred_color = np.concatenate((pred_x_color, pred_y_color), axis=2)
    fl2.visualize_flow(cv2.resize(flo_pred_color, dsize=(300, 200),
               interpolation=cv2.INTER_LINEAR), maxrad)

    pred_x_refine_color = pred_x_refine[:, :, np.newaxis]
    pred_y_refine_color = pred_y_refine.T[:, :, np.newaxis]
    flo_pred_refine_color = np.concatenate((pred_x_refine_color, pred_y_refine_color), axis=2)
    fl2.visualize_flow(cv2.resize(flo_pred_refine_color, dsize=(300, 200),
               interpolation=cv2.INTER_LINEAR), maxrad)


    # pred_x_vis = cv2.resize(vis_RGB(pred_x, np.min(flow_x), np.max(flow_x)), dsize=(300, 200),
    #            interpolation=cv2.INTER_LINEAR)
    # pred_y_vis = cv2.resize(vis_RGB(pred_y.T, np.min(flow_y), np.max(flow_y)), dsize=(300, 200),
    #            interpolation=cv2.INTER_LINEAR)

    # cv2.imshow("name", pred_x_vis)
    # cv2.waitKey(0)
    #
    # cv2.imshow("name", pred_y_vis)
    # cv2.waitKey(0)