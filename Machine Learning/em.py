import numpy as np
import matplotlib.pyplot as plt
import random
from scipy.stats import multivariate_normal


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

    data = np.row_stack((first_gauss, second_gauss, third_gauss))
    # print(data)
    print(len(data))
    return data


class EM(object):
    def __init__(self, data):
        self.data = data
        # 设定初始化的参数
        self.mu = [[random.uniform(min(data[:, 0]), max(data[:, 0])),
                    random.uniform(min(data[:, 1]), max(data[:, 1]))],
                   [random.uniform(min(data[:, 0]), max(data[:, 0])),
                    random.uniform(min(data[:, 1]), max(data[:, 1]))],
                   [random.uniform(min(data[:, 0]), max(data[:, 0])),
                    random.uniform(min(data[:, 1]), max(data[:, 1]))]]
        self.cov = [np.identity(2), np.identity(2), np.identity(2)]
        # 初始化隐变量z，分别表示属于属于每个guass的概率
        rand_array = np.random.rand(3)
        self.z = rand_array / np.sum(rand_array)
        self.epsilon = 0.001

    # Expectation Step
    def expectation(self):
        alpha = [0.0, 0.0, 0.0]
        for i in range(3):
            alpha[i] = multivariate_normal.pdf(self.data, self.mu[i], self.cov[i])

        beta = np.zeros((3, len(self.data)))
        for i in range(3):
            beta[i] = self.z[i] * alpha[i] / (self.z[0] * alpha[0] + self.z[1] * alpha[1] + self.z[2] * alpha[2])

        return beta

    # Maximization Step
    def maximization(self, beta):
        # 根据E步所推断的出来的z值，来求最优化的各个参数
        for i in range(3):
            beta_sum = np.sum(beta[i])
            self.mu[i] = np.dot(beta[i], self.data) / beta_sum
            self.cov[i] = np.dot(beta[i] * (self.data - self.mu[i]).T, self.data - self.mu[i]) / beta_sum
            self.z[i] = beta_sum / len(self.data)

    # EM算法
    def em(self):
        k = 0
        end = True
        print('初始行均值,', 'mu1:', self.mu[0], 'mu2:', self.mu[1], 'mu3:', self.mu[2])
        while end:
            self.draw_plot(k)
            k += 1
            end = False
            old_mu1 = self.mu[0].copy()
            old_mu2 = self.mu[1].copy()
            old_mu3 = self.mu[2].copy()
            # 先通过E步求隐变量的后验分布
            beta = self.expectation()
            # 然后通过M步最大化各个参数
            self.maximization(beta)
            print(k, '行均值,', 'mu1:', self.mu[0], 'mu2:', self.mu[1], 'mu3:', self.mu[2])
            for i in range(2):
                if abs(self.mu[0][i] - old_mu1[i]) > self.epsilon or abs(self.mu[1][i] - old_mu2[i]) > self.epsilon or abs(
                        self.mu[2][i] - old_mu3[i]) > self.epsilon:
                    end = True
                    break

        print('类别概率:\tgauss1:', self.z[0], '\tgauss2:', self.z[1], '\tgauss3:', self.z[2])
        print('均值:\tmu1:', self.mu[0], '\tmu2:', self.mu[1], '\tmu3:', self.mu[2])
        print('协方差矩阵:\ncov1:', self.cov[0], '\ncov2:', self.cov[1], '\ncov3:', self.cov[2])

    # 对当前的分类结果进行绘图
    def draw_plot(self, index):
        gauss1 = multivariate_normal(self.mu[0], self.cov[0])
        gauss2 = multivariate_normal(self.mu[1], self.cov[1])
        gauss3 = multivariate_normal(self.mu[2], self.cov[2])
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

        plt.scatter(self.mu[0][0], self.mu[0][1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu[0][0], 2)) + ',' + str(round(self.mu[0][1], 2)) + ')',
                     xy=(self.mu[0][0], self.mu[0][1]), fontsize=20)
        plt.scatter(self.mu[1][0], self.mu[1][1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu[1][0], 2)) + ',' + str(round(self.mu[1][1], 2)) + ')',
                     xy=(self.mu[1][0], self.mu[1][1]), fontsize=20)
        plt.scatter(self.mu[2][0], self.mu[2][1], s=40, edgecolors='black')
        plt.annotate('(' + str(round(self.mu[2][0], 2)) + ',' + str(round(self.mu[2][1], 2)) + ')',
                     xy=(self.mu[2][0], self.mu[2][1]), fontsize=20)
        plt.grid()
        # plt.savefig('em_res/cluster_res' + str(index) + '.jpg')
        plt.show()


if __name__ == '__main__':
    # 初始化3个二元的正态分布
    train_data = data_generation([0, 0], np.identity(2), [3, 3], np.identity(2), [-3, 5], np.identity(2))
    em = EM(train_data)
    em.em()
