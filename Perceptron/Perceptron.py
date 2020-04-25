class Perceptron(object):
    def __init__(self, c, _train_data, _train_label, _initial_weight):
        self.c = c          # c：权向量校正时候的校正增量
        self.train_data = _train_data   # train_data：所有的模式
        self.train_label = _train_label # train_label：所有模式所属的类别
        self.weight = _initial_weight   # initial_weight：权向量的初始化权重
        self.dimension = len(train_data[0])
        if len(_train_data) != len(_train_label) or self.dimension != len(_initial_weight) - 1:
            print("数据长度有误")
            exit()

    def data_normalization(self):
        # 对数据进行规范化处理，也就是负类的样本全部乘以-1
        for label, data in zip(self.train_label, self.train_data):
            if label == -1:
                for index, item in enumerate(data):
                    data[index] = - data[index]

    def result_for_x(self, x, index):
        # x表示一个特定的模式，对x分类，返回分类结果
        res = 0
        for i in range(self.dimension):
            res += self.weight[i] * x[i]
        res += self.weight[self.dimension] * self.train_label[index]
        return res

    def update_weight(self):
        # 根据每一个训练数据，更新权重
        end = True
        for index, data in enumerate(self.train_data):
            # 预测当前训练数据的结果
            res = self.result_for_x(data, index)
            # 如果分类错误，则更新权重
            if res <= 0:
                end = False
                for i in range(self.dimension):
                    self.weight[i] += self.c * data[i]
                self.weight[self.dimension] += self.train_label[index]
                # 记录预测结果，并记录更新后的权重值
                print('第' + str(index+1) + '个数据错判，修改权重当前权重为', str(self.weight))
            else:
                print('第' + str(index+1) + '个数据判别正确')
        return end

    def get_res_weight(self):
        self.data_normalization()
        i = 1
        while True:
            res = self.update_weight()
            if res:
                print("感知器算法结束，最终结果权重为：", str(self.weight))
                return
            else:
                print("@@@@@@@@@@@@@@@@@@@@@第" + str(i) + "轮迭代结果：", str(self.weight))
            i += 1


if __name__ == '__main__':
    train_data = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]]
    train_label = [1, 1, 1, 1, -1, -1, -1, -1]
    initial_weight = [-1, -2, -2, 0]
    perceptron = Perceptron(2, train_data, train_label, initial_weight)
    perceptron.get_res_weight()
