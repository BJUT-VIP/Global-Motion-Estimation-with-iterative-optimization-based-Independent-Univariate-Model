"""
created on 2020/3/27
function: 生成光流数据的全局运动参数，作为基于神经网络的全局运动估计算法的 gt
"""

from lib import flowlib as fl
from lib import flowlib_v2 as fl2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2
import glob
from natsort import natsorted
import pickle
from scipy import optimize

# hyper parameter of the threshold function, the bigger the hyper the less point
# (less points will be considered as local motion)
hyper = 1.0


def fitting_funcA(x, a, b):
    return a * x + b


def fitting_funcB(x, a, b, c):
    return a * x[1] + b * x[0] + c


def LeastSquareFit(flo_data, index_list):
    fita_list = []
    fita, fitb = optimize.curve_fit(fitting_funcA, index_list, flo_data)
    fita_list.append(fita)
    fita_avg = np.mean(np.array(fita_list), axis=0)
    return fita_avg


def aux(data):
    i, lst = data[0], data[1]
    if i not in lst:
        return i[1]


def param_estimate_x(flo_data, index_list, outlier_mat, iteration):
    outline_index_list = []
    fita_avg = None
    for iter in range(iteration):
        # x_axis_list 里面存的是所有点的x坐标值
        # data 里面存的是每个点对应的运动位移值，已归一化到了[0, 255]
        x_axis_list = np.array([i[1] for i in index_list if i not in outline_index_list])
        data = np.array([flo_data[i[0], i[1], 0] for i in index_list if i not in outline_index_list])

        ## Model A
        fita_avg = LeastSquareFit(data, x_axis_list)
        data_new = np.arange(0, len(flo_data[0]))
        data_com = fita_avg[0] * data_new.reshape(1, -1) + fita_avg[1]

        ## Model B
        # x_axis_square = np.square(x_axis_list)
        # fita_avg = LeastSquareFit(data, [x_axis_list, x_axis_square])
        # data_new = np.arange(0, len(flo_data[0]))
        # data_com = fita_avg[0] * data_new.reshape(1, -1) ** 2 + fita_avg[1] * data_new.reshape(1, -1) + fita_avg[2]

        thres_auto = (np.max(data_com) - np.min(data_com)) * (hyper / pow(2, iter))
        # thres_auto = abs(data_com[0][-1] - data_com[0][0]) * (hyper / pow(2, iter))

        if thres_auto < 0.1:
            thres_auto = 0.1

        x_mat = np.repeat(data_com, flo_data.shape[0], axis=0)
        m_diff = np.abs(flo_data[:, :, 0] - x_mat)

        # mask 是一个和原始光流场大小一样的矩阵，里面值是true和false。用来指明哪些是异常点
        mask = np.where(m_diff > thres_auto, True, False)

        outlier_mat = outlier_mat | mask
        outline_index_list = np.argwhere(outlier_mat).tolist()

        index_set = set(index_list)
        index_set.difference_update(set(map(tuple, outline_index_list)))
        index_list = list(index_set)

    return fita_avg


def param_estimate_y(flo_data, index_list, outlier_mat, iteration):
    outline_index_list = []
    fita_avg = None
    for iter in range(iteration):
        # y_axis_list 里面存的是所有点的y坐标值
        # data 里面存的是每个点对应的运动位移值，已归一化到了[0, 255]
        y_axis_list = np.array([i[0] for i in index_list if i not in outline_index_list])

        data = np.array([flo_data[i[0], i[1], 1] for i in index_list if i not in outline_index_list])

        ## model A
        fita_avg = LeastSquareFit(data, y_axis_list)
        data_new = np.arange(0, len(flo_data))
        data_com = fita_avg[0] * data_new.reshape(-1, 1) + fita_avg[1]

        ## model B
        # y_axis_square = np.square(y_axis_list)
        # fita_avg = LeastSquareFit(data, [y_axis_list, y_axis_square])
        # data_new = np.arange(0, len(flo_data))
        # data_com = fita_avg[0] * data_new.reshape(-1, 1) ** 2 + fita_avg[1] * data_new.reshape(-1, 1) + fita_avg[2]

        thres_auto = (np.max(data_com) - np.min(data_com)) * (hyper / pow(2, iter))
        # thres_auto = abs(data_com[0][-1] - data_com[0][0]) * (hyper / pow(2, iter))

        if thres_auto < 0.1:
            thres_auto = 0.1

        # print('y channel iter:{}, auto_thres:{}'.format(iter, thres_auto))

        y_mat = np.repeat(data_com, flo_data.shape[1], axis=1)

        m_diff = np.abs(flo_data[:, :, 1] - y_mat)

        # mask 是一个和原始光流场大小一样的矩阵，里面值是true和false。用来指明哪些是异常点
        mask = np.where(m_diff > thres_auto, True, False)
        outlier_mat = outlier_mat | mask
        outline_index_list = np.argwhere(outlier_mat).tolist()
        index_set = set(index_list)
        index_set.difference_update(set(map(tuple, outline_index_list)))
        index_list = list(index_set)

    return fita_avg


def global_estimate(flo_data):

    # fl.visualize_flow(flo_data)
    displace = 15
    iteration = 3

    flo_data = np.clip(flo_data, -displace, displace)
    index_list = [(i, j) for i in range(0, flo_data.shape[0]) for j in range(0, flo_data.shape[1])]
    outlier_mat = np.full((flo_data.shape[0], flo_data.shape[1]), False)

    x_para = param_estimate_x(flo_data, index_list, outlier_mat, iteration=iteration)
    y_para = param_estimate_y(flo_data, index_list, outlier_mat, iteration=iteration)

    return x_para, y_para


def reconstruct(data, maxrad, w=200, h=200):
    x_data = data[0]
    y_data = data[1]
    line = np.arange(0, w)
    data_com_x = x_data[0] * line.reshape(1, -1) ** 2 + x_data[1] * line.reshape(1, -1) + x_data[2]
    x_mat = np.repeat(data_com_x, h, axis=0)

    col = np.arange(0, h)
    data_com_y = y_data[0] * col.reshape(-1, 1) ** 2 + y_data[1] * col.reshape(-1, 1) + y_data[2]

    y_mat = np.repeat(data_com_y, w, axis=1)

    flo_x = x_mat[:, :, np.newaxis]
    flo_y = y_mat[:, :, np.newaxis]
    flo_global = np.concatenate((flo_x, flo_y), axis=2)
    flo_global_color = fl2.flow_to_image(flo_global, maxrad)
    plt.imshow(flo_global_color)
    plt.show()


if __name__ == '__main__':

    # with open('D:/TMM_exp/global_param_gtA_ucf.pkl', 'rb') as f:
    #     res_ucf = pickle.load(f)
    # with open('D:/TMM_exp/global_param_gtA_ncaa.pkl', 'rb') as f:
    #     res_ncaa = pickle.load(f)
    # res_ncaa.update(res_ucf)
    # with open('D:/TMM_exp/global_param_gtA_all.pkl', 'wb') as f:
    #     pickle.dump(res_ncaa, f)

    base_dir = 'D:/TMM_exp/test'

    # with open('D:/TMM_exp/global_param_gt.pkl', 'rb') as f:
    #     res = pickle.load(f)
    res = {}

    event_list = os.listdir(base_dir)
    for event in event_list:
        print(event)
        time_list = os.listdir(os.path.join(base_dir, event))
        for time_point in time_list:
            print(time_point)
            flo_list = natsorted(glob.glob(os.path.join(base_dir, event, time_point) + '/*.npy'))
            for flo_path in flo_list:
                print(flo_path)
                res[flo_path] = []
                data_org = np.load(flo_path).astype(np.float32)
                # data_org = fl.read_flo_file(flo_path)
                data = cv2.resize(data_org, dsize=(64, 64), interpolation=cv2.INTER_LINEAR)
                # fl.visualize_flow(data)
                # u = data[:, :, 0]
                # v = data[:, :, 1]
                # rad = np.sqrt(u ** 2 + v ** 2)  # 光流场方向
                # maxrad = max(-1, np.max(rad))

                para = list(global_estimate(data))
                # reconstruct(para, maxrad)
                res[flo_path].append(para)
    with open('D:/TMM_exp/global_param_gtA_test.pkl', 'wb') as f:
        pickle.dump(res, f)
    pass