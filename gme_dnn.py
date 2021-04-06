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


def img_process(data_dict, path):
    flo_data = np.load(path).astype(np.float32)
    data = cv2.resize(flo_data, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
    channel = random.randint(0, 1)
    index = random.randint(0, 63)
    param = data_dict[path][0][channel]
    # label = np.multiply(label, np.array([1000, 10, 1]))
    data_new = np.arange(0, 64)
    # label = param[0] * data_new.reshape(1, -1) + param[1]
    label = param[0] * data_new.reshape(1, -1) ** 2 + param[1] * data_new.reshape(1, -1) + param[2]
    if channel == 0:
        data_rand = data[index:index + 1, :, 0]
    else:
        data_rand = data[:, index:index + 1, 1].reshape(1, -1)
    # if data_rand.var() > 0.1:
    #     # print('invalid data')
    #     return None, label
    return data_rand, label


def gen_data_label(data_dict, path_list, batch_size):
    data_set = []
    label_set = []
    for i in range(1, len(path_list)):
        data, label = img_process(data_dict, path_list[i])
        # if data is None:
        #     # print('invalid data drop')
        #     continue
        # img_data_array = np.asarray(img_data_pil)
        # data = torch.from_numpy(img_data_array)
        data_set.append(data)
        label_set.append(label)
        d = np.array(data_set)

        if i % batch_size == 0:
            yield torch.tensor(d).type(torch.FloatTensor), torch.tensor(label_set).type(torch.FloatTensor)
            data_set = []
            label_set = []


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with open('D:/TMM_exp/global_param_gtB_all.pkl', 'rb') as f:
        data_dict = pickle.load(f)
        path_list = list(data_dict.keys())

    batch_size = 64
    # net = NetA(64, 128, 64)
    net = NetB(64, 128, 128, 64)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    net.train()
    net.to(device)

    loss_list = []

    for epoch in range(300):
        random.shuffle(path_list)
        for index, data in enumerate(gen_data_label(data_dict, path_list, batch_size)):
            # print(index)
            inputs = data[0].to(device)
            labels = data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            loss_list.append(float(loss))

            if index % 20 == 0:
                print(loss.item())
                print(labels[0])
                print(outputs[0])

    x_arr = list(np.arange(0, len(loss_list)))
    plt.plot(x_arr, loss_list, marker='.', color='r', label='time')
    plt.show()

    with open('./loss/loss_netB_modelB.pkl', 'wb') as f:
        pickle.dump(loss_list, f)

    torch.save(net.state_dict(), './model/netB_modelB_epoc300.pkl')