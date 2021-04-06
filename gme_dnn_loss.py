import matplotlib.pyplot as plt
import pickle
import numpy as np

if __name__ == '__main__':
    with open('./loss/loss_netB_modelB.pkl', 'rb') as f:
        loss_list = pickle.load(f)
    loss_list = loss_list[0:1500]

    x_axis = np.arange(0, len(loss_list))
    plt.plot(x_axis, loss_list, 'r')
    plt.show()