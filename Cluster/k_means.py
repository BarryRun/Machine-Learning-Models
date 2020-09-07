from collections import defaultdict
import random
from math import sqrt
import numpy as np
import matplotlib.pyplot as plt


class Cluster(object):
    def __init__(self, points, k):
        self.points = points        # 所有样本点
        self.dimensions = len(points[0])    # 样本的维度
        self.centers = []           # 每个类别的中心点
        self.center_num = k         # 类别的个数
        self.point_cluster = []     # 表示每个点所属的类别

    # 使用k_means算法对样本点进行分类
    def k_means(self):
        # 随机产生k个中心点
        self.initial_k_centers()
        # 为每个点进行分类
        self.assign_points()

        old_assignments = None

        # 直到聚类结果不再发生变化，结束聚类
        epoch = 0
        while self.point_cluster != old_assignments:
            # 重新计算每个类别的中心点
            self.update_centers()
            old_assignments = self.point_cluster
            # 为每个点进行分类
            self.assign_points()
            self.draw_pic(epoch)
            epoch += 1

    # 画出样本点，并为不用样本点分别着色
    def draw_pic(self, epoch):
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        # 对于所有点以及其所属类别
        for cluster, point in zip(self.point_cluster, self.points):
            # 将不同类别的点分开
            plt.plot(point[0], point[1], mark[cluster], markersize=5)
        plt.savefig('images/k_means'+str(epoch)+'.jpg')
        plt.show()

    # 计算样本的均值
    def point_avg(self, points):
        # 获取特征的维度
        centers = []

        # 计算每个维度上面所有点对应值的和
        for dimension in range(self.dimensions):
            dim_sum = 0  # dimension sum
            for p in points:
                dim_sum += float(p[dimension])

            # 计算每个维度上点的平均值
            centers.append(dim_sum / float(len(points)))

        return centers

    # 给定样本及各样本点所属的类别，计算新的聚类中心点
    def update_centers(self):
        # 定义一个字典，其默认初始化为一个空列表
        new_means = defaultdict(list)
        centers = []

        # 对于所有点以及其所属类别
        for cluster, point in zip(self.point_cluster, self.points):
            # 将不同类别的点分开
            new_means[cluster].append(point)

        # 对于每个类别的所有点
        for points in new_means.values():
            # 通过point_avg计算其中心点
            centers.append(self.point_avg(points))

        self.centers =  centers

    # 给定样本以及聚类中心，为每个样本点计算最近的聚类中心
    def assign_points(self):
        point_cluster = []

        # 对于数据中的每一个点
        for point in self.points:
            # float("inf")表示正无穷，加上负号，即float("-inf")为负无穷
            shortest = float("inf")  # positive infinity
            shortest_index = 0
            # 对于每一个中心
            for i in range(self.center_num):
                # 计算到每一个中心的距离
                val = self.euclidean_distance(point, self.centers[i])

                # 记录距离最短的中心距离，以及其index
                if val < shortest:
                    shortest = val
                    shortest_index = i
            point_cluster.append(shortest_index)
        # 返回一个数组，对应data_points中相同index的点类别（即距离最近的中心点）
        self.point_cluster =  point_cluster

    # 计算欧氏距离
    def euclidean_distance(self, a, b):
        _sum = 0
        for dimension in range(self.dimensions):
            difference_sq = (a[dimension] - b[dimension]) ** 2
            _sum += difference_sq
        return sqrt(_sum)

    # 给定样本，初始时随机生成k个聚类中心
    def initial_k_centers(self):
        data_num = len(self.points)
        # 如果样本点数量小于k，则无法进行聚类
        if data_num < self.center_num:
            return None

        # 随机选择k个点为初始的聚类点
        random_ints = random.sample(range(0, data_num-1), self.center_num)
        centers = []
        for i in range(self.center_num):
            centers.append(self.points[random_ints[i]])

        self.centers = centers


if __name__ == '__main__':
    test_points = [[0, 0], [3, 8], [2, 2], [1, 1], [5, 3], [4, 8], [6, 3], [5, 4], [6, 4], [7, 5]]
    first_gauss = np.random.multivariate_normal([0, 2], np.identity(2), 20, check_valid="raise")
    second_gauss = np.random.multivariate_normal([3, 3], np.identity(2), 20, check_valid="raise")
    third_gauss = np.random.multivariate_normal([-1, 5], np.identity(2), 20, check_valid="raise")
    data = np.row_stack((first_gauss, second_gauss, third_gauss))
    cluster = Cluster(data, 3)
    cluster.k_means()
