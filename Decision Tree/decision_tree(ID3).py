# -- coding: utf-8 --
import pandas as pd
import math
import pickle
import random
from collections import defaultdict

def data_loader(data_path):
    """
    本文所使用的数据来源于 https://archive.ics.uci.edu/ml/datasets/Parkinson%27s+Disease+Classification#
    根据一些特征，来判断目标是否患有帕金森
    :param data_path: path of the data
    :return:
    """
    csv_data = pd.read_csv(data_path)
    # 由于大部分特征都是连续值，因此需要做离散化
    # 将所有连续特征都分箱成五个部分
    for coloumn in csv_data.columns:
        if coloumn not in ['id', 'class', 'gender']:
            csv_data[coloumn] = pd.cut(csv_data[coloumn], 5, labels=False)

    # 划分训练集、测试集
    train_data = csv_data.sample(frac=0.8, random_state=4396, axis=0)
    test_data = csv_data[~csv_data.index.isin(train_data.index)]
    # 划分特征与标签
    train_y = train_data['class']
    train_x = train_data.drop(columns=['id', 'class'])
    test_y = test_data['class']
    test_x = test_data.drop(columns=['id', 'class'])

    # 该数据没有缺失值，因此不需要进行缺失值的处理
    return train_x, train_y, test_x, test_y


def decision_tree(train_x, train_y):
    """
    决策树算法整体框架
    :return:
    """
    # 1.创建决策树
    # 如果样本全部属于同一类别，则直接返回类别
    label_set = set()
    for label in train_y:
        label_set.add(label)
    if len(label_set) == 1:
        return label_set.pop()

    # 如果属性集合为空，则返回数据集中占大多数的类别
    columns = list(train_x.columns)
    if not columns:
        return major_label(train_y)

    # 在当前的特征集合中，选取最优的特征
    best_feature, vals_for_bes_feature = select_best_feature(train_x, train_y)

    # 根据该特征的每一个值，生成一个分支
    tree = {best_feature:{}}
    for val in vals_for_bes_feature:
        # 删除非对应值的数据, 同时删除对应特征列
        index_to_del = train_x[train_x[best_feature] != val].index
        tree[best_feature][val] = decision_tree(train_x.drop(index_to_del).drop(columns=[best_feature]),
                                                train_y.drop(index_to_del).drop(columns=[best_feature]))
    print(tree)
    return tree


def predict(tree, test_x, test_y):
    print(test_x)
    positive = 0.
    negative = 0.
    for (idx1, row1), label in zip(test_x.iterrows(), test_y):
    # for x, y in zip(test_x, test_y):
        # 根据当前特征的值，来决定走哪一个分支
        key = list(tree.keys())[0]
        # print(key, row1)
        next = tree[key][row1[key]]
        while type(next) is dict:
            print(next)
            key = list(next.keys())[0]
            # 这里有可能会产生缺失值问题。即：测试集中的特征未曾在训练集中出现过
            # 这里先随机地分到一类中去，后续再进行补充处理
            if row1[key] not in next[key].keys():
                random_key = random.sample(next[key].keys(), 1)[0]
                next = next[key][random_key]
            else:
                next = next[key][row1[key]]
        if next == label:
            positive += 1
        else:
            negative += 1
    print('Acc is:', positive/(positive+negative))


def select_best_feature(_x, _y):
    """
    由于在前面所有特征均已经过离散化处理，因此把所有特征都当做离散特征来进行处理
    为了方便起见，这里采用了计算信息增益的方法来进行决策树的划分属性选择
    """
    features = _x.columns
    vals_for_best_features = {}
    best_feature = ''
    best_info_gain = 0.0
    entropy = calculate_entropy(_y)
    # 对所有特征进行遍历
    for feature in features:
        # 获取当前feature下所有可能的取值
        feature_set = set(_x[feature])
        # 对于每一个可能的取值，划分数据集并计算信息熵
        new_entropy = 0
        for a_feature in feature_set:
            # 根据所选择的特征值来筛选出子数据集
            index_to_drop = _x[_x[feature] != a_feature].index
            sub_y = _y.drop(index_to_drop)
            # 计算信息增益
            weight = len(sub_y) / len(_y)
            new_entropy += weight * calculate_entropy(sub_y)
        # 计算选取当前特征的信息增益
        info_gain = entropy - new_entropy
        if info_gain > best_info_gain:
            best_info_gain = info_gain
            best_feature = feature
            vals_for_best_features = feature_set
            # print(feature, info_gain, vals_for_best_features)
    return best_feature, vals_for_best_features


def calculate_entropy(labels):
    """
    计算当前数据集下的信息熵
    """
    # 计算所有label出现的次数
    label_count = defaultdict(float)
    for label in labels:
        label_count[label] += 1

    res = 0
    # 计算所有label出现的频率
    for label in label_count:
        ratio = label_count[label] / len(labels)
        res -= ratio*(math.log(ratio, 2))

    return res


def major_label(labels):
    """
    统计一组标签中数据量最大的一种标签
    :param labels: 标签序列
    :return:
    """
    label_num = defaultdict(int)
    for label in labels:
        label_num[label] += 1
    label_num = sorted(label_num.items(), key=lambda x:x[1], reverse=True)
    return label_num[0][0]



if __name__ == '__main__':
    x, y, t_x, t_y = data_loader('pd_speech_features.csv')
    # tree = decision_tree(x, y)
    # pickle.dump(tree, open("model_tree.pk", 'wb'))
    predict(pickle.load(open('model_tree.pk', 'rb')), t_x, t_y)
