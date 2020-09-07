import numpy as np
import pandas as pd
import math
import csv


# 此处数据由李宏毅老师的深度学习课程提供
# http://speech.ee.ntu.edu.tw/~tlkagk/courses_ML20.html
# 数据预处理部分，参考 https://colab.research.google.com/drive/131sSqmrmWXfjFZ3jWSELl8cm0Ox5ah3C
def data_preprocess():
    data = pd.read_csv('train.csv', encoding='big5')
    data = data.iloc[:, 3:]
    data[data == 'NR'] = 0
    raw_data = data.to_numpy()
    # 数据是12个月，每个月为开始的20天数据，每天有24个小时，每个小时有18个feature，即18列
    # 考虑将连续天的数据拼接起来，最终应该是12个月*18个feature即216行，24个小时*20天即480列，最终数据为12*18*480
    data_processed = {}
    for month in range(12):
        buf = np.empty([18, 480])  # 代表一个月的数据
        for day in range(20):
            buf[:, day * 24:(day + 1) * 24] = raw_data[(month * 20 + day) * 18:(month * 20 + day + 1) * 18, :]
        data_processed[month] = buf

    # 经过上述处理，每个月有480个小时，每10个小时都可以作为一次训练数据（因为测试数据就是从9个小时预测1个小时）
    # 因此共有12个月*471条训练数据，每条数据长度为18个特征*9个小时， label长度为1
    # 这里需要注意我们单独的一个训练数据可以是一个18*9的矩阵，也可以是长度为18*9的一个向量，这里作为向量输入
    train_x = np.empty([12 * 471, 18 * 9], dtype=float)
    train_y = np.empty([12 * 471, 1], dtype=float)
    for month in range(12):
        for day in range(20):
            for hour in range(24):
                # 每个月只有20天的数据
                if day == 19 and hour > 14:
                    continue
                # 这里的reshape指定了行数为1，相当于将原本18行*9列的数据变成了一行，即1*162，表示有162个特征
                train_x[month * 471 + day * 24 + hour, :] = \
                    data_processed[month][:, day * 24 + hour: day * 24 + hour + 9].reshape(1, -1)
                # 这里的9是指PM2.5数据为第9个特征
                train_y[month * 471 + day * 24 + hour, 0] = data_processed[month][9, day * 24 + hour + 9]  # value
    print(len(train_x[0]))

    # 对数据进行标准化 Normalization，这里用的是z-score标准化，使得数据符合标准的正态分布
    # 这里axis=0表示忽略第一个维度（行），对其余维度所有数据做对应操作
    # 求每一列数据的平均与方差
    mean_x = np.mean(train_x, axis=0)
    std_x = np.std(train_x, axis=0)
    for i in range(len(train_x)):  # 12 * 471
        for j in range(len(train_x[0])):  # 18 * 9
            if std_x[j] != 0:
                train_x[i][j] = (train_x[i][j] - mean_x[j]) / std_x[j]

    # 对数据进行shuffle后，划分训练集与验证集合
    # 通过get_state()保存状态，set_state()重新载入状态，使得shuffle过程保持对应关系不变
    state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(state)
    np.random.shuffle(train_y)
    # 或者直接通过sklearn进行shuffle也可以
    # x, y = shuffle(train_x, train_y)

    x_train_set = train_x[: math.floor(len(train_x) * 0.8), :]
    y_train_set = train_y[: math.floor(len(train_y) * 0.8), :]
    x_validation = train_x[math.floor(len(train_x) * 0.8):, :]
    y_validation = train_y[math.floor(len(train_y) * 0.8):, :]

    return x_train_set, y_train_set, x_validation, y_validation, mean_x, std_x


def linear_regression(x_train_set, y_train_set, x_validation, y_validation):
    # 首先根据训练数据长度，声明参数w，是包含b的列向量，其中+1表示bias
    feature_num = len(x_train_set[0])
    w = np.empty([feature_num + 1, 1])
    # 给特征加上一个1，用来与b相乘
    x_train_set = np.concatenate((np.ones([len(x_train_set), 1]), x_train_set), axis=1).astype(float)
    x_validation = np.concatenate((np.ones([len(x_validation), 1]), x_validation), axis=1).astype(float)
    # 设置相关的超参数，例如学习率
    learning_rate = 10

    # 一直到收敛的时候才停止训练
    last_val_loss = 0   # 记录上一轮训练的loss值
    epoch = 0   # 统计训练轮数
    stay_epoch = 0   # 统计多少轮训练loss未发生变化
    adagrad = np.zeros([feature_num + 1, 1])    # 采用ada梯度下降算法，为每一个feature分配一个学习率
    eps = 0.0000000001  # 防止除0
    while True:
        # 计算均方根误差 Root Mean Square Error
        loss = np.sqrt(np.sum(np.power(np.dot(x_train_set, w) - y_train_set, 2))/len(x_train_set))
        val_loss = np.sqrt(np.sum(np.power(np.dot(x_validation, w) - y_validation, 2))/len(x_validation))
        # 每训练50轮打印一次loss
        if epoch % 100 == 0:
            print(str(epoch) + ' epoch: train loss=' + str(loss) + ' val_loss=' + str(val_loss))

        # 判断是否停止迭代：如果val_loss超过20轮迭代未发生下降，则停止训练
        if val_loss >= last_val_loss or last_val_loss - val_loss <= 10e-6:
            stay_epoch += 1
            if stay_epoch == 20:
                print(str(epoch) + ' epoch训练结束: train loss=' + str(loss) + ' val_loss=' + str(val_loss))
                break
        else:
            stay_epoch = 0

        epoch += 1
        # 梯度下降，计算下降的梯度：这里需要对均方根误差进行求导
        gradient = 2 * np.dot(x_train_set.transpose(), np.dot(x_train_set, w) - y_train_set)  # dim*1
        # adagrad迭代中公式，可参照 https://www.jianshu.com/p/a8637d1bb3fc， 即梯度越大，学习率越小
        adagrad += gradient ** 2
        w = w - learning_rate * gradient / np.sqrt(adagrad + eps)   # 加eps防止除0
        last_val_loss = val_loss

    # 保存训练模型
    np.save('weight.npy', w)


def predict(mean_x, std_x):
    test_data = pd.read_csv('test.csv', header=None, encoding='big5')
    test_data = test_data.iloc[:, 2:]
    test_data[test_data == 'NR'] = 0
    test_data = test_data.to_numpy()
    test_x = np.empty([240, 18 * 9], dtype=float)
    # 与训练数据相似的处理过程：将数据进行排列
    for i in range(240):
        test_x[i, :] = test_data[18 * i: 18 * (i + 1), :].reshape(1, -1)

    # 同样对数据进行归一化，这里需要重新读取训练数据的规则化参数
    for i in range(len(test_x)):
        for j in range(len(test_x[0])):
            if std_x[j] != 0:
                test_x[i][j] = (test_x[i][j] - mean_x[j]) / std_x[j]
    test_x = np.concatenate((np.ones([240, 1]), test_x), axis=1).astype(float)

    w = np.load('weight.npy')
    ans_y = np.dot(test_x, w)

    with open('submit.csv', mode='w', newline='') as submit_file:
        csv_writer = csv.writer(submit_file)
        header = ['id', 'value']
        # print(header)
        csv_writer.writerow(header)
        for i in range(240):
            row = ['id_' + str(i), ans_y[i][0]]
            csv_writer.writerow(row)
            # print(row)


if __name__ == '__main__':
    x_train, y_train, x_val, y_val, mean_x, std_x = data_preprocess()
    linear_regression(x_train, y_train, x_val, y_val)
    predict(mean_x, std_x)
