# -- coding: utf-8 --
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import math


# 随机生成由3个二元的正态分布所组成的数据, 二维的更加直观
def data_generation(mu1, cov1, mu2, cov2, mu3, cov3):
    first_gauss = np.random.multivariate_normal(mu1, cov1, 200, check_valid="raise")
    second_gauss = np.random.multivariate_normal(mu2, cov2, 200, check_valid="raise")
    third_gauss = np.random.multivariate_normal(mu3, cov3, 200, check_valid="raise")

    # 绘制出数据的分布
    plots1 = plt.scatter(first_gauss[:, 0], first_gauss[:, 1], s=5, edgecolors='red')
    plots2 = plt.scatter(second_gauss[:, 0], second_gauss[:, 1], s=5, edgecolors='blue')
    plots3 = plt.scatter(third_gauss[:, 0], third_gauss[:, 1], s=5, edgecolors='green')
    plt.legend([plots1, plots2, plots3], ['first gauss', 'second gauss', 'third guass'], loc='upper right')
    plt.grid()
    plt.show()

    # 添加label，这里的方法有点蠢
    first_gauss = np.pad(first_gauss, pad_width=1, mode='constant', constant_values=0)[1:-1, 1:]
    second_gauss = np.pad(second_gauss, pad_width=1, mode='constant', constant_values=1)[1:-1, 1:]
    third_gauss = np.pad(third_gauss, pad_width=1, mode='constant', constant_values=2)[1:-1, 1:]

    data = np.row_stack((first_gauss, second_gauss, third_gauss))
    # print(data)
    # print(len(data))
    return data


def nb(all_data):
    # 打乱数据，然后划分为训练集与测试集
    np.random.shuffle(all_data)
    train_data = all_data[:200]
    test_data = all_data[200:]
    train_length = len(train_data)
    count1 = count2 = count0 = 0
    train_data1 = []
    train_data2 = []
    train_data0 = []

    # 首先需要计算先验概率
    for i in range(train_length):
        label = int(train_data[i, 2])
        if label == 0:
            train_data0.append(train_data[i])
            count0 += 1
        elif label == 1:
            train_data1.append(train_data[i])
            count1 += 1
        else:
            train_data2.append(train_data[i])
            count2 += 1

    # print(np.array(train_data0))
    prior_prob0 = count0 / train_length
    prior_prob1 = count1 / train_length
    prior_prob2 = count2 / train_length

    # 然后需要计算输入x的每一个维度下，对于不同类别的概率分布
    # 在离散情况下直接通过计算频数即可，连续情况下则需要假设其分布，这里假设为高斯分布（实际上也就是高斯分布(*^▽^*)
    # 计算平均值与标准差
    train_data = [np.array(train_data0), np.array(train_data1), np.array(train_data2)]
    mean = np.ones(shape=(3, 2))
    std = np.ones(shape=(3, 2))
    for i in range(3):
        for j in range(2):
            mean[i, j] = np.mean(train_data[i][:, j])
            std[i, j] = np.std(train_data[i][:, j])

    # ---------------------这样就求出来了先验概率与条件概率，接下来就可以进行预测了--------------------
    right = 0
    wrong = 0
    # 通过高斯分布来计算每个参数的概率
    for i in range(len(test_data)):
        prob = [prior_prob0, prior_prob1, prior_prob2]
        for j in range(3):
            for k in range(2):
                prob[j] = prob[j] * gaussian_probability(test_data[i][k], mean[j][k], std[j][k])
        if prob.index(max(prob)) == int(test_data[i][2]):
            right += 1
        else:
            wrong += 1
    print('预测的准确率为：', right/len(test_data))


def gaussian_probability(x, mean, std):
    exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(std, 2))))
    return (1 / (math.sqrt(2 * math.pi) * std)) * exponent


if __name__ == '__main__':
    data = data_generation([0, 0], np.identity(2), [3, 3], np.identity(2), [-3, 5], np.identity(2))
    nb(data)