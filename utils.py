import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np



def generate_target_mnist(x, y, normal_index, anomaly_ratio, concentrated=False):
    if concentrated is False:
        ind_4 = np.where(y == normal_index)
        ind_other_digit = np.where(y != normal_index)
        x_normal = x[ind_4, :, :].squeeze()
        x_anomaly = x[ind_other_digit, :, :].squeeze()
        n_anomaly = x_anomaly.shape[0]
        n_normal = x_normal.shape[0]

        perm = torch.randperm(n_anomaly)
        n_anomaly = int(n_normal * anomaly_ratio)
        selected_anomaly = perm[:n_anomaly]
        x_anomaly = x_anomaly[selected_anomaly, :, :]
        x_train = np.concatenate((x_normal, x_anomaly), 0)
        # x = x.view(x.shape[0], -1)
        y_train = np.zeros(n_normal + n_anomaly)
        y_train[n_normal:] = 1

        x_test = x.squeeze()
        y_test = y != normal_index
        return x_train, y_train, x_test, y_test

    else:
        ind_4 = np.where(y == normal_index)
        ind_other_digit = np.where(y == (normal_index + 1)%10)
        # ind_other_digit = np.where(y != normal_index)
        x_normal = x[ind_4, :, :].squeeze()
        x_anomaly = x[ind_other_digit, :, :].squeeze()
        n_anomaly = x_anomaly.shape[0]
        n_normal = x_normal.shape[0]

        perm = torch.randperm(n_anomaly)
        n_anomaly = int(n_normal * anomaly_ratio)
        selected_anomaly = perm[:n_anomaly]
        x_anomaly = x_anomaly[selected_anomaly, :, :]
        x_train = np.concatenate((x_normal, x_anomaly), 0)
        # x = x.view(x.shape[0], -1)
        y_train = np.zeros(n_normal + n_anomaly)
        y_train[n_normal:] = 1

        x_test = x.squeeze()
        y_test = y != normal_index
        return x_train, y_train, x_test, y_test


def generate_target_cifar10(x, y, normal_index, anomaly_ratio, concentrated = False):
    if concentrated is False:
        ind_4 = np.where(y == normal_index)
        ind_other_digit = np.where(y != normal_index)
        x_normal = x[ind_4, :, :, :].squeeze()
        x_anomaly = x[ind_other_digit, :, :, :].squeeze()
        n_anomaly = x_anomaly.shape[0]
        n_normal = x_normal.shape[0]

        perm = torch.randperm(n_anomaly)
        n_anomaly = int(n_normal * anomaly_ratio)
        selected_anomaly = perm[:n_anomaly]
        x_anomaly = x_anomaly[selected_anomaly, :, :, :]
        x_train = np.concatenate((x_normal, x_anomaly), 0)
        # x = x.view(x.shape[0], -1)
        y_train = np.zeros(n_normal + n_anomaly)
        y_train[n_normal:] = 1

        x_test = x
        y_test = y != normal_index
        return x_train, y_train, x_test, y_test
    else:
        ind_4 = np.where(y == normal_index)
        ind_other_digit = np.where(y == (normal_index + 1)%10)
        x_normal = x[ind_4, :, :, :].squeeze()
        x_anomaly = x[ind_other_digit, :, :, :].squeeze()

        n_anomaly = x_anomaly.shape[0]
        n_normal = x_normal.shape[0]


        perm = torch.randperm(n_anomaly)
        n_anomaly = int(n_normal * anomaly_ratio)
        selected_anomaly = perm[:n_anomaly]
        x_anomaly = x_anomaly[selected_anomaly, :, :, :]
        x_train = np.concatenate((x_normal, x_anomaly), 0)

        # x = x.view(x.shape[0], -1)
        y_train = np.zeros(n_normal + n_anomaly)
        y_train[n_normal:] = 1

        x_test = np.concatenate((x_normal, x[ind_other_digit, :, :, :].squeeze()), 0)
        y_test = np.zeros(x_test.shape[0])
        y_test[n_normal:]=1
        return x_train, y_train, x_test, y_test


def generate_target_cifar10_concentrated(xtrn, ytrn, xtst, ytst, normal_index, anomaly_ratio):
        ind_4 = np.where(ytrn != normal_index)
        ind_other_digit = np.where(ytrn == (normal_index))
        x_normal = xtrn[ind_4, :, :, :].squeeze()
        x_anomaly = xtrn[ind_other_digit, :, :, :].squeeze()

        n_anomaly = x_anomaly.shape[0]
        n_normal = x_normal.shape[0]

        perm = torch.randperm(n_anomaly)
        n_anomaly = int(n_normal * anomaly_ratio)
        selected_anomaly = perm[:n_anomaly]
        x_anomaly = x_anomaly[selected_anomaly, :, :, :]
        x_train = np.concatenate((x_normal, x_anomaly), 0)

        y_train = np.zeros(n_normal + n_anomaly)
        y_train[n_normal:] = 1


        ind_other_digit = np.where(ytst == (normal_index))

        x_test = xtst
        y_test = np.zeros(xtst.shape[0])
        y_test[ind_other_digit] = 1
        return x_train, y_train, x_test, y_test



def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class Shuffling(nn.Module):
    def __init__(self,):
        super(Shuffling, self).__init__()

    def forward(self, z):
        n, d = z.shape
        z_shuffle = z.copy_()
        # shuffling
        for i in range(d):
            z_shuffle[:, i] = z[torch.randperm(n), i]
        return z_shuffle
