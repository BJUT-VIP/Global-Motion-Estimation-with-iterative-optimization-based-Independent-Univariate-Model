"""
使用最小二乘拟合的方法对全局运动进行估计，通过不断迭代消除局部运动的干扰 2019/12/04
"""

from lib import flowlib_v2 as fl2
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from lib import flowlib as fl
from scipy import optimize
import time
from math import pow

# hyper parameter of the threshold function, the bigger the hyper the less point
# (less points will be considered as local motion)
hyper = 1.0


def fitting_func(x, a, b, c):
    return a * x[0] + b * x[1] + c


def fitting_func2(x, a, b):
    return a * x + b


def LeastSquareFit(flo_data, index_list):
    """
        Parameters
    ----------
    f : callable
        The model function, f(x, ...).  It must take the independent
        variable as the first argument and the parameters to fit as
        separate remaining arguments.
    xdata : An M-length sequence or an (k,M)-shaped array for functions with k predictors
        The independent variable where the data is measured.
    ydata : M-length sequence
        The dependent data --- nominally f(xdata, ...)
    """
    v_y = index_list
    v_z = flo_data
    fita_list = []

    fita, fitb = optimize.curve_fit(fitting_func, np.array(v_y), np.array(v_z))
    fita_list.append(fita)
    fita_avg = np.mean(np.array(fita_list), axis=0)
    return fita_avg


def aux(data):
    i, lst = data[0], data[1]
    if i not in lst:
        return i[1]


def return_path(base_path, depth, index_list):
    sub_path = base_path
    for i in range(depth):
        file_list = sorted(os.listdir(sub_path))
        sub_path = os.path.join(sub_path, file_list[index_list[i]])

    return sub_path


def param_estimate_u(flo_data, index_list):
    """
    index_list: {(h, w)}
    """
    # x_axis_list 里面存的是所有点的x坐标值
    # data 里面存的是每个点对应的运动位移值
    ux_axis_list = [i[1] for i in index_list]
    uy_axis_list = [i[0] for i in index_list]
    data = [flo_data[i[0], i[1], 0] for i in index_list]
    fita_avg = LeastSquareFit(data, [ux_axis_list, uy_axis_list])

    data_com = np.zeros((flo_data.shape[0], flo_data.shape[1]))
    for i in range(flo_data.shape[0]):
        for j in range(flo_data.shape[1]):
            data_com[i][j] = fita_avg[0] * j + fita_avg[1] * i + fita_avg[2]
    return data_com


def param_estimate_y(flo_data, index_list):

    # y_axis_list 里面存的是所有点的y坐标值
    # data 里面存的是每个点对应的运动位移值
    vx_axis_list = [i[1] for i in index_list]
    vy_axis_list = [i[0] for i in index_list]
    data = [flo_data[i[0], i[1], 1] for i in index_list]

    fita_avg = LeastSquareFit(data, [vx_axis_list, vy_axis_list])

    data_com = np.zeros((flo_data.shape[0], flo_data.shape[1]))
    for i in range(flo_data.shape[0]):
        for j in range(flo_data.shape[1]):
            data_com[i][j] = fita_avg[0] * j + fita_avg[1] * i + fita_avg[2]
    return data_com


def global_motion_estimation(flo_data, w=490, h=360):
    # mask = np.where(np.abs(flo_data) < 0.8, 0, 1)
    # mask = mask[:, :, 0] & mask[:, :, 1]
    # mask = np.expand_dims(mask, axis=2)
    # mask = np.repeat(mask, 2, axis=2)

    displace = 15
    # thres = (np.max(flo_data) - np.min(flo_data)) * 2
    # print(thres)

    # u = flo_data[:, :, 0]
    # v = flo_data[:, :, 1]
    # rad = np.sqrt(u ** 2 + v ** 2)  # 光流场方向
    # maxrad = max(-1, np.max(rad))

    flo_data = np.clip(flo_data, -displace, displace)
    index_list = [(i, j) for i in range(0, flo_data.shape[0]) for j in range(0, flo_data.shape[1])]

    x_ch = param_estimate_u(flo_data, index_list)
    y_ch = param_estimate_y(flo_data, index_list)

    flo_x = x_ch[:, :, np.newaxis]
    flo_y = y_ch[:, :, np.newaxis]
    flo_global = np.concatenate((flo_x, flo_y), axis=2)

    flo_global = cv2.resize(flo_global, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

    return flo_global


if __name__ == '__main__':
    file_name = r'D:\学习资料\TMM2019\TMM_exp\ncaa\3-pointer-failure\173806.535\9.npy'
    print(file_name)

    start = time.time()

    # path = '/data3/yz/dataset/NCAA/flo_data/3gtm0aaBkxM/other-2-pointer-failure/1417272.537/10.npy'

    # path = '/data3/yz/dataset/UCF-101/UCF-101-flo/BalanceBeam/v_BalanceBeam_g15_c02/2.npy'

    data_org = np.load(file_name).astype(np.float32)

    # fl.visualize_flow(data_org)

    print('org_data_size: {}'.format(data_org.shape))

    resize_w = 64  # ncaa 128, ucf 32
    resize_h = 36  # ncaa 72, ucf 24

    data = cv2.resize(data_org, dsize=(resize_w, resize_h), interpolation=cv2.INTER_LINEAR)
    global_motion = global_motion_estimation(data, w=data_org.shape[1], h=data_org.shape[0])
    end = time.time()
    print('time: ', end - start)

    epe = fl.flow_error(global_motion[0], global_motion[1], data_org[0], data_org[1])
    print('epe: {}'.format(epe))

    local_motion = data_org - global_motion

    local_motion_outlier = local_motion * outlier_set

    flo_global_color = fl2.flow_to_image(global_motion, maxrad)
    flo_local_color = fl2.flow_to_image(local_motion, maxrad)

    local_motion_outlier_rad = np.sqrt(local_motion_outlier[:, :, 0] ** 2 + local_motion_outlier[:, :, 1] ** 2)
    local_motion_outlier_mask = np.where(np.abs(local_motion_outlier_rad) > 1.0, 1, 0)

    local_motion_outlier[:, :, 0] *= local_motion_outlier_mask
    local_motion_outlier[:, :, 1] *= local_motion_outlier_mask

    flo_local_outlier_color = fl2.flow_to_image(local_motion_outlier, maxrad)

    fl.visualize_flow(data_org)

    plt.imshow(flo_global_color)
    plt.show()

    plt.imshow(flo_local_color)
    plt.show()

    plt.imshow(flo_local_outlier_color)
    plt.show()
