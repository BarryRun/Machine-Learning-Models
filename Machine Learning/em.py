import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# 随机生成两个二元的正态分布, 二维的更加直观
def data_generation(mu1, cov1, alpha1, mu2, cov2, alpha2):
    first_guass = np.random.multivariate_normal(mu1, cov1, 100, check_valid="raise")
    second_guass = np.random.multivariate_normal(mu2, cov2, 100, check_valid="raise")

    # 绘制出数据的分布
    plots1 = plt.scatter(first_guass[:, 0], first_guass[:, 1], s=5)
    plots2 = plt.scatter(second_guass[:, 0], second_guass[:, 1], s=5)
    plt.legend([plots1, plots2], ['first gauss', 'second gauss'], loc='upper right')
    plt.grid()
    plt.show()

    data = np.row_stack((first_guass, second_guass))
    # print(data)
    print(len(data))
    label = [0] * 100 + [1] * 100
    return data, label


class EM(object):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        # 设定初始化的参数, 分别表示：
        # 第一个gauss的均值、协方差，第二个gauss的均值、协方差，以及属于第一个gauss概率
        self.mu1 = [1, 1]
        self.cov1 = np.identity(2)
        self.mu2 = [2, 2]
        self.cov2 = np.identity(2)
        # 隐变量z，表示数据属于第一个gauss的概率
        self.z = 0.5
        self.epsilon = 0.00001

    # Expectation Step
    def expectation(self):
        # 这里的隐变量z看做每个数据所属于的高斯类别
        # 概率密度函数，即计算每个数据在当前分布下出现的概率
        alpha1 = self.z * multivariate_normal.pdf(self.data, self.mu1, self.cov1)
        alpha2 = (1 - self.z) * multivariate_normal.pdf(self.data, self.mu2, self.cov2)
        # beta表示每个数据出现在第一个gauss下的概率
        # 每个样本属于第一个gauss的概率
        beta = alpha1 / (alpha1 + alpha2)
        return beta

    # Maximization Step
    def maximization(self, beta):
        # 根据E步所推断的出来的z值，来求最优化的各个参数
        self.mu1 = np.dot(beta, self.data) / np.sum(beta)
        self.mu2 = np.dot((1 - beta), self.data) / np.sum(1 - beta)
        self.cov1 = np.dot(beta * (self.data - self.mu1).T, self.data - self.mu1) / np.sum(beta)
        self.cov2 = np.dot((1 - beta) * (self.data - self.mu2).T, self.data - self.mu2) / np.sum(1 - beta)
        self.z = np.sum(beta) / 200

    # EM算法
    def em(self):
        k = 0
        end = True
        while end:
            k += 1
            end = False
            old_mu1 = self.mu1.copy()
            old_mu2 = self.mu2.copy()
            beta = self.expectation()
            self.maximization(beta)
            print(k, '行均值,', 'mu1:', self.mu1, 'mu2:', self.mu2)
            for i in range(2):
                if not ((self.mu1[i] - old_mu1[i]) < self.epsilon and (self.mu2[i] - old_mu2[i]) < self.epsilon):
                    end = True
                    break

        print('类别概率:\t', self.z)
        print('均值:\t', self.mu1, self.mu2)
        print('方差:\n', self.cov1, '\n', self.cov2, '\n')


if __name__ == '__main__':
    # 初始化两个二元的正态分布，并各随机生成100个点
    # 两个正态分布的参数分别如下
    # 1、均值为[0, 0]，协方差矩阵为[[1, 0], [0, 1]]，即各维度方差为1，且各维度之间不相关
    # 2、均值为[3, 3]，协方差矩阵为[[1, 0], [0, 1]]，同上
    train_data, train_label = data_generation([0, 0], np.identity(2), 0.4, [3, 3], np.identity(2), 0.6)
    em = EM(train_data, train_label)
    em.em()
