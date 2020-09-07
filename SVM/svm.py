import numpy as np


# 首先尝试最简单的数据线性可分的硬间隔SVM
class SVM:
    def __init__(self, c, max_epoch, data_num=1000):
        # 用于初始化一些超参数
        self.data_num = data_num
        self.max_epoch = max_epoch
        self.C = c

    def linear_data_construct(self):
        """
        生成一组线性可分的数据
        数据分布于以(0,0) (0,1) (1,0) (1,1)为顶点的正方形内，位于直线 y=x上方则类别为1，
        :return: x,y: 分别代表坐标、类别
        """
        # 随机生成100个0-1之间的二元组
        x = np.random.rand(self.data_num, 2)
        y = []
        for a in x:
            if a[1] > a[0]:
                y.append(1)
            else:
                y.append(-1)
        return x, y

    def first_to_opt(self, x, y):

    def second_to_opt


    def fit_smo(self, x, y):
        """
        具体的计算方法参考 https://zhuanlan.zhihu.com/p/140912728
        :param x: 训练数据的特征
        :param y: 训练数据的标签
        :return:
        """
        epoch = 0
        alpha = [0] * self.data_num
        for epoch in range(self.max_epoch):
            # 选择第一个优化变量
            idx1 = first_to_opt


if __name__ == '__main__':
    svm = SVM()
    svm.linear_data_construct()
