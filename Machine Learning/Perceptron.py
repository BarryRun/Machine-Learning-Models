class Perceptron(object):
    def __init__(self, c, _train_data, _train_label, _initial_weight):
        # c：权向量校正时候的校正增量
        # train_data：所有的模式
        # train_label：所有模式所属的类别
        # initial_weight：权向量的初始化权重
        self.c = c
        self.train_data = _train_data
        self.train_label = _train_label
        self.weight = _initial_weight
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
        # x表示一个特定的模式，对x进行加权求解，如果结果小于0，则修改权重
        res = 0
        for i in range(self.dimension):
            res += self.weight[i] * x[i]
        res += self.weight[self.dimension] * self.train_label[index]
        return res

    def update_weight(self):
        end = True
        for index, data in enumerate(self.train_data):
            res = self.result_for_x(data, index)
            if res <= 0:
                print('第' + str(index) + '个数据错判，修改权重')
                end = False
                for i in range(self.dimension):
                    self.weight[i] += self.c * data[i]
                self.weight[self.dimension] += self.train_label[index]
                print("当前权重为", str(self.weight))
            else:
                print('第' + str(index) + '个数据判别正确')
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
            # if i == 10:
            #     break


if __name__ == '__main__':
    train_data = [[0, 0, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 1]]
    train_label = [1, 1, 1, 1, -1, -1, -1, -1]
    initial_weight = [-1, -2, -2, 0]
    perceptron = Perceptron(1, train_data, train_label, initial_weight)
    perceptron.get_res_weight()