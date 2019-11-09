import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal


# 随机生成两个二元的正态分布, 二维的更加直观
def data_generation(mu1, cov1, mu2, cov2, mu3, cov3):
    first_gauss = np.random.multivariate_normal(mu1, cov1, 300, check_valid="raise")
    second_gauss = np.random.multivariate_normal(mu2, cov2, 300, check_valid="raise")
    third_gauss = np.random.multivariate_normal(mu3, cov3, 300, check_valid="raise")

    # 绘制出数据的分布
    plots1 = plt.scatter(first_gauss[:, 0], first_gauss[:, 1], s=5, edgecolors='red')
    plots2 = plt.scatter(second_gauss[:, 0], second_gauss[:, 1], s=5, edgecolors='blue')
    plots3 = plt.scatter(third_gauss[:, 0], third_gauss[:, 1], s=5, edgecolors='green')
    plt.legend([plots1, plots2, plots3], ['first gauss', 'second gauss', 'third guass'], loc='upper right')
    plt.grid()
    plt.show()

    data = np.row_stack((first_gauss, second_gauss, third_gauss))
    # print(data)
    print(len(data))
    return data


class EM(object):
    def __init__(self, data):
        self.data = data
        # 设定初始化的参数, 分别表示：
        # 第一个gauss的均值、协方差，第二个gauss的均值、协方差
        self.mu1 = [random.uniform(min(data[:, 0]), max(data[:, 0])),
                    random.uniform(min(data[:, 1]), max(data[:, 1]))]
        self.cov1 = np.identity(2)
        self.mu2 = [random.uniform(min(data[:, 0]), max(data[:, 0])),
                    random.uniform(min(data[:, 1]), max(data[:, 1]))]
        self.cov2 = np.identity(2)
        self.mu3 = [random.uniform(min(data[:, 0]), max(data[:, 0])),
                    random.uniform(min(data[:, 1]), max(data[:, 1]))]
        self.cov3 = np.identity(2)
        # 初始化隐变量z，分别表示属于属于每个guass的概率
        rand_array = np.random.rand(3)
        self.z = rand_array / np.sum(rand_array)
        self.epsilon = 0.001

    # Expectation Step
    def expectation(self):
        # 这里的隐变量z看做每个数据所属于的高斯类别
        # 概率密度函数，即计算每个数据在当前分布下出现的概率
        # alpha1表示gauss1中数据出现的概率
        # alpha2表示gauss2中数据出现的概率
        alpha1 = multivariate_normal.pdf(self.data, self.mu1, self.cov1)
        alpha2 = multivariate_normal.pdf(self.data, self.mu2, self.cov2)
        alpha3 = multivariate_normal.pdf(self.data, self.mu3, self.cov3)
        # beta表示每个样本属于不同gauss的概率
        beta = np.zeros((3, len(self.data)))
        beta[0] = self.z[0] * alpha1 / (self.z[0] * alpha1 + self.z[1] * alpha2 + self.z[2] * alpha3)
        beta[1] = self.z[1] * alpha2 / (self.z[0] * alpha1 + self.z[1] * alpha2 + self.z[2] * alpha3)
        beta[2] = self.z[2] * alpha3 / (self.z[0] * alpha1 + self.z[1] * alpha2 + self.z[2] * alpha3)
        return beta

    # Maximization Step
    def maximization(self, beta):
        # 根据E步所推断的出来的z值，来求最优化的各个参数
        self.mu1 = np.dot(beta[0], self.data) / np.sum(beta[0])
        self.mu2 = np.dot(beta[1], self.data) / np.sum(beta[1])
        self.mu3 = np.dot(beta[2], self.data) / np.sum(beta[2])
        self.cov1 = np.dot(beta[0] * (self.data - self.mu1).T, self.data - self.mu1) / np.sum(beta[0])
        self.cov2 = np.dot(beta[1] * (self.data - self.mu2).T, self.data - self.mu2) / np.sum(beta[1])
        self.cov3 = np.dot(beta[2] * (self.data - self.mu2).T, self.data - self.mu2) / np.sum(beta[2])
        self.z[0] = np.sum(beta[0]) / len(self.data)
        self.z[1] = np.sum(beta[1]) / len(self.data)
        self.z[2] = np.sum(beta[2]) / len(self.data)

    # EM算法
    def em(self):
        k = 0
        end = True
        print('初始行均值,', 'mu1:', self.mu1, 'mu2:', self.mu2, 'mu3:', self.mu3)
        while end:
            self.draw_plot(k)
            k += 1
            end = False
            old_mu1 = self.mu1.copy()
            old_mu2 = self.mu2.copy()
            old_mu3 = self.mu3.copy()
            # 先通过E步求隐变量的后验分布
            beta = self.expectation()
            # 然后通过M步最大化各个参数
            self.maximization(beta)
            print(k, '行均值,', 'mu1:', self.mu1, 'mu2:', self.mu2, 'mu3:', self.mu3)
            for i in range(2):
                if abs(self.mu1[i] - old_mu1[i]) > self.epsilon or abs(self.mu2[i] - old_mu2[i]) > self.epsilon or abs(
                        self.mu3[i] - old_mu3[i]) > self.epsilon:
                    end = True
                    break

        print('类别概率:\tgauss1:', self.z[0], '\tgauss2:', self.z[1], '\tgauss3:', self.z[2])
        print('均值:\tmu1:', self.mu1, '\tmu2:', self.mu2, '\tmu3:', self.mu3)
        print('协方差矩阵:\ncov1:', self.cov1, '\ncov2:', self.cov2, '\ncov3:', self.cov3)

    # 对当前的分类结果进行绘图
    def draw_plot(self, index):
        gauss1 = multivariate_normal(self.mu1, self.cov1)
        gauss2 = multivariate_normal(self.mu2, self.cov2)
        gauss3 = multivariate_normal(self.mu3, self.cov3)
        p1 = gauss1.pdf(self.data)
        p2 = gauss2.pdf(self.data)
        p3 = gauss3.pdf(self.data)
        for i in range(len(self.data)):
            if max(p1[i], p2[i], p3[i]) == p1[i]:
                plt.scatter(self.data[i][0], self.data[i][1], s=5, edgecolors='red')
            elif max(p1[i], p2[i], p3[i]) == p2[i]:
                plt.scatter(self.data[i][0], self.data[i][1], s=5, edgecolors='blue')
            else:
                plt.scatter(self.data[i][0], self.data[i][1], s=5, edgecolors='green')

        plt.scatter(self.mu1[0], self.mu1[1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu1[0], 2)) + ',' + str(round(self.mu1[1], 2)) + ')',
                     xy=(self.mu1[0], self.mu1[1]), fontsize=20)
        plt.scatter(self.mu2[0], self.mu2[1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu2[0], 2)) + ',' + str(round(self.mu2[1], 2)) + ')',
                     xy=(self.mu2[0], self.mu2[1]), fontsize=20)
        plt.scatter(self.mu3[0], self.mu3[1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu3[0], 2)) + ',' + str(round(self.mu3[1], 2)) + ')',
                     xy=(self.mu3[0], self.mu3[1]), fontsize=20)
        plt.grid()
        # plt.savefig('em_res/cluster_res' + str(index) + '.jpg')
        plt.show()


if __name__ == '__main__':
    # 初始化两个二元的正态分布，并各随机生成100个点
    # 两个正态分布的参数分别如下
    # 1、均值为[0, 0]，协方差矩阵为[[1, 0], [0, 1]]，即各维度方差为1，且各维度之间不相关
    # 2、均值为[3, 3]，协方差矩阵为[[1, 0], [0, 1]]，同上
    train_data = data_generation([0, 0], np.identity(2), [3, 3], np.identity(2), [-3, 5], np.identity(2))
    em = EM(train_data)
    em.em()
