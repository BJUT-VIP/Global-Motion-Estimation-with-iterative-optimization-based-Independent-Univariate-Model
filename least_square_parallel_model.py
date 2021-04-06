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


def fitting_func(x, a, b):
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
    fita_list = []

    fita, fitb = optimize.curve_fit(fitting_func, index_list, flo_data)
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


def param_estimate_u(flo_data, index_list, outlier_mat, iteration):
    """
    index_list: {(h, w)}
    """
    outline_index_list = []
    x_mat = None
    thres_auto = 0.0
    for iter in range(iteration):
        # x_axis_list 里面存的是所有点的x坐标值
        # data 里面存的是每个点对应的运动位移值
        ux_axis_list = np.array([i[1] for i in index_list if i not in outline_index_list])
        # uy_axis_list = [i[0] for i in index_list if i not in outline_index_list]
        # uy_aixs_list = [i[0] for i in index_list if i not in outline_index_list]
        data = np.array([flo_data[i[0], i[1], 0] for i in index_list if i not in outline_index_list])

        # plot remaining points
        # plt.scatter(ux_axis_list, data, marker='.', c='y')

        # fita_avg 里面是拟合出来的参数值
        fita_avg = LeastSquareFit(data, ux_axis_list)

        data_new = np.arange(0, len(flo_data[0]))
        data_com = fita_avg[0] * data_new.reshape(1, -1) + fita_avg[1]

        thres_auto = abs(data_com[0][-1] - data_com[0][0]) * (hyper / pow(2, iter))
        # print('x channel iter:{}, auto_thres:{}'.format(iter, thres_auto))

        x_mat = np.repeat(data_com, flo_data.shape[0], axis=0)
        m_diff = np.abs(flo_data[:, :, 0] - x_mat)

        # mask 是一个和原始光流场大小一样的矩阵，里面值是true和false。用来指明哪些是异常点
        mask = np.where(m_diff > thres_auto, True, False)

        outlier_mat = outlier_mat | mask
        outline_index_list = np.argwhere(outlier_mat).tolist()

        index_set = set(index_list)
        index_set.difference_update(set(map(tuple, outline_index_list)))
        index_list = list(index_set)

        # plot function curve
        # plt.ylim(-5, 5)
        # plt.title("x: iter {}".format(iter))
        # plt.plot(np.array(data_new), np.squeeze(data_com), linewidth=5)
        # plt.show()

    return x_mat, outlier_mat


def param_estimate_y(flo_data, index_list, outlier_mat, iteration):
    outline_index_list = []
    y_mat = None
    thres_auto = 0.0
    for iter in range(iteration):
        # y_axis_list 里面存的是所有点的y坐标值
        # data 里面存的是每个点对应的运动位移值
        vx_axis_list = np.array([i[0] for i in index_list if i not in outline_index_list])
        # vy_axis_list = [i[0] for i in index_list if i not in outline_index_list]
        data = np.array([flo_data[i[0], i[1], 1] for i in index_list if i not in outline_index_list])

        # plot remaining points
        # plt.scatter(vx_axis_list, data, marker='.', c='y')

        # fita_avg 里面是拟合出来的参数值
        # start1 = time.time()
        fita_avg = LeastSquareFit(data, vx_axis_list)
        # end1 = time.time()
        # print('fitting time: ', end1-start1)

        data_new = np.arange(0, len(flo_data))
        data_com = fita_avg[0] * data_new.reshape(-1, 1) + fita_avg[1]

        thres_auto = abs(data_com[0][0] - data_com[-1][0]) * (hyper / pow(2, iter))

        y_mat = np.repeat(data_com, flo_data.shape[1], axis=1)

        m_diff = np.abs(flo_data[:, :, 1] - y_mat)

        # mask 是一个和原始光流场大小一样的矩阵，里面值是true和false。用来指明哪些是异常点
        mask = np.where(m_diff > thres_auto, True, False)
        outlier_mat = outlier_mat | mask
        outline_index_list = np.argwhere(outlier_mat).tolist()
        index_set = set(index_list)
        index_set.difference_update(set(map(tuple, outline_index_list)))
        index_list = list(index_set)

        # plot function curve
        # plt.ylim(-5, 8)
        # plt.title("y: iter {}".format(iter))
        # plt.plot(np.array(data_new), np.squeeze(data_com), linewidth=5)
        # plt.show()

    return y_mat, outlier_mat


def global_motion_estimation(flo_data, w=490, h=360):
    # mask = np.where(np.abs(flo_data) < 0.8, 0, 1)
    # mask = mask[:, :, 0] & mask[:, :, 1]
    # mask = np.expand_dims(mask, axis=2)
    # mask = np.repeat(mask, 2, axis=2)

    displace = 15
    iteration = 1
    # thres = (np.max(flo_data) - np.min(flo_data)) * 2
    # print(thres)

    u = flo_data[:, :, 0]
    v = flo_data[:, :, 1]
    rad = np.sqrt(u ** 2 + v ** 2)  # 光流场方向
    maxrad = max(-1, np.max(rad))

    flo_data = np.clip(flo_data, -displace, displace)
    index_list = [(i, j) for i in range(0, flo_data.shape[0]) for j in range(0, flo_data.shape[1])]
    outlier_mat = np.full((flo_data.shape[0], flo_data.shape[1]), False)

    x_ch, outlier_x = param_estimate_u(flo_data, index_list, outlier_mat, iteration=iteration)
    y_ch, outlier_y = param_estimate_y(flo_data, index_list, outlier_mat, iteration=iteration)

    outlier_merge = np.expand_dims(outlier_x | outlier_y, axis=2)
    outlier_merge = np.repeat(outlier_merge, 2, axis=2)
    outlier_merge = cv2.resize(outlier_merge.astype(np.float32), dsize=(w, h), interpolation=cv2.INTER_LINEAR)

    flo_x = x_ch[:, :, np.newaxis]
    flo_y = y_ch[:, :, np.newaxis]
    flo_global = np.concatenate((flo_x, flo_y), axis=2)

    flo_global = cv2.resize(flo_global, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

    # flo_local = flo_data - flo_global
    #
    # flo_global_color = fl2.flow_to_image(flo_global, maxrad)
    # flo_local_color = fl2.flow_to_image(flo_local, maxrad)

    return flo_global, maxrad, outlier_merge


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
    global_motion, maxrad, outlier_set = global_motion_estimation(data, w=data_org.shape[1], h=data_org.shape[0])
    end = time.time()
    print('time: ', end - start)

    epe = fl.flow_error(global_motion[0], global_motion[1], data_org[0], data_org[1])
    print('epe: {}'.format(epe))

    # local_motion = data_org - global_motion
    #
    # local_motion_outlier = local_motion * outlier_set
    #
    # flo_global_color = fl2.flow_to_image(global_motion, maxrad)
    # flo_local_color = fl2.flow_to_image(local_motion, maxrad)
    #
    # local_motion_outlier_rad = np.sqrt(local_motion_outlier[:, :, 0] ** 2 + local_motion_outlier[:, :, 1] ** 2)
    # local_motion_outlier_mask = np.where(np.abs(local_motion_outlier_rad) > 1.0, 1, 0)
    #
    # local_motion_outlier[:, :, 0] *= local_motion_outlier_mask
    # local_motion_outlier[:, :, 1] *= local_motion_outlier_mask
    #
    # flo_local_outlier_color = fl2.flow_to_image(local_motion_outlier, maxrad)
    #
    # fl.visualize_flow(data_org)
    #
    # plt.imshow(flo_global_color)
    # plt.show()
    #
    # plt.imshow(flo_local_color)
    # plt.show()
    #
    # plt.imshow(flo_local_outlier_color)
    # plt.show()
