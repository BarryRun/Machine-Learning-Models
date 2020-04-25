import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# 随机生成两个二元的正态分布, 二维的更加直观
def data_generation(mu1, cov1, mu2, cov2):
    first_guass = np.random.multivariate_normal(mu1, cov1, 1000, check_valid="raise")
    second_guass = np.random.multivariate_normal(mu2, cov2, 1000, check_valid="raise")

    # 绘制出数据的分布
    plots1 = plt.scatter(first_guass[:, 0], first_guass[:, 1], s=5)
    plots2 = plt.scatter(second_guass[:, 0], second_guass[:, 1], s=5)
    plt.legend([plots1, plots2], ['first gauss', 'second gauss'], loc='upper right')
    plt.grid()
    plt.show()

    data = np.row_stack((first_guass, second_guass))
    # data = alpha1 * first_guass + alpha2 * second_guass
    print(data)
    print(len(data))
    return data


class EM(object):
    def __init__(self, data):
        self.data = data
        # 设定初始化的参数, 分别表示：
        # 第一个gauss的均值、协方差，第二个gauss的均值、协方差
        self.mu1 = [1, 1]
        self.cov1 = np.identity(2)
        self.mu2 = [5, 5]
        self.cov2 = np.identity(2)
        # 隐变量z，表示数据属于第一个gauss的概率
        self.z = 0.5
        self.epsilon = 0.001

    # Expectation Step
    def expectation(self):
        # 这里的隐变量z看做每个数据所属于的高斯类别
        # 概率密度函数，即计算每个数据在当前分布下出现的概率
        # alpha1表示gauss1中数据出现的概率
        # alpha2表示gauss2中数据出现的概率
        alpha1 = multivariate_normal.pdf(self.data, self.mu1, self.cov1)
        alpha2 = multivariate_normal.pdf(self.data, self.mu2, self.cov2)
        # beta表示每个样本属于第一个gauss的概率
        beta = self.z * alpha1 / (self.z * alpha1 + (1 - self.z) * alpha2)
        return beta

    # Maximization Step
    def maximization(self, beta):
        # 根据E步所推断的出来的z值，来求最优化的各个参数
        self.mu1 = np.dot(beta, self.data) / np.sum(beta)
        self.mu2 = np.dot((1 - beta), self.data) / np.sum(1 - beta)
        self.cov1 = np.dot(beta * (self.data - self.mu1).T, self.data - self.mu1) / np.sum(beta)
        self.cov2 = np.dot((1 - beta) * (self.data - self.mu2).T, self.data - self.mu2) / np.sum(1 - beta)
        self.z = np.sum(beta) / len(self.data)

    # EM算法
    def em(self):
        k = 0
        end = True
        print(k, '行均值,', 'mu1:', self.mu1, 'mu2:', self.mu2)
        while end:
            self.draw_plot(k)
            k += 1
            end = False
            old_mu1 = self.mu1.copy()
            old_mu2 = self.mu2.copy()
            beta = self.expectation()
            self.maximization(beta)
            print(k, '行均值,', 'mu1:', self.mu1, 'mu2:', self.mu2)
            for i in range(2):
                if abs(self.mu1[i] - old_mu1[i]) > self.epsilon or abs(self.mu2[i] - old_mu2[i]) > self.epsilon:
                    end = True
                    break

        print('类别概率:\tgauss1:', self.z, '\tgauss2:', 1 - self.z)
        print('均值:\tmu1:', self.mu1, '\tmu2:', self.mu2)
        print('协方差矩阵:\ncov1:', self.cov1, '\ncov2:', self.cov2, '\n')

    # 对当前的分类结果进行绘图
    def draw_plot(self, index):
        gauss1 = multivariate_normal(self.mu1, self.cov1)
        gauss2 = multivariate_normal(self.mu2, self.cov2)
        p1 = gauss1.pdf(self.data)
        p2 = gauss2.pdf(self.data)
        for i in range(len(self.data)):
            if p1[i] > p2[i]:
                plt.scatter(self.data[i][0], self.data[i][1], s=5, edgecolors='red')
            else:
                plt.scatter(self.data[i][0], self.data[i][1], s=5, edgecolors='blue')
        plt.scatter(self.mu1[0], self.mu1[1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu1[0], 2)) + ',' + str(round(self.mu1[1], 2)) + ')',
                     xy=(self.mu1[0], self.mu1[1]), fontsize=20)
        plt.scatter(self.mu2[0], self.mu2[1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu2[0], 2)) + ',' + str(round(self.mu2[1], 2)) + ')',
                     xy=(self.mu2[0], self.mu2[1]), fontsize=20)
        plt.grid()
        # plt.savefig('simple_em_res/cluster_res' + str(index) + '.jpg')
        plt.show()


if __name__ == '__main__':
    # 初始化两个二元的正态分布，并各随机生成100个点
    # 两个正态分布的参数分别如下
    # 1、均值为[0, 0]，协方差矩阵为[[1, 0], [0, 1]]，即各维度方差为1，且各维度之间不相关
    # 2、均值为[3, 3]，协方差矩阵为[[1, 0], [0, 1]]，同上
    train_data= data_generation([0, 0], np.identity(2), [3, 3], np.identity(2))
    em = EM(train_data)
    em.em()
