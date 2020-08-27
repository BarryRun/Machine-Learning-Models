import numpy as np
import matplotlib.pyplot as plt

# 首先从文件中加载数据
# 数据来源于 https://github.com/csuldw/MachineLearning/blob/master/PCA/data.txt
data = []
with open('data.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data.append([float(val) for val in line.split()])
data = np.array(data)

# 去中心化
data -= np.mean(data, axis=0)

# 计算协方差矩阵
cov = np.cov(data, rowvar=False)

# 特征分解，得到特征向量和特征值
w, v = np.linalg.eig(cov)
print(v)

# 选出最大的两个特征值与其对应的特征向量
sorted_indices = np.argsort(w)[::-1][:2]
print(v[:, sorted_indices])

# 将原样本点投影到选取的特征向量上
data = np.dot(data, v[:, sorted_indices])

# 作图查看降维效果
xVal = data[:, 0]
yVal = -data[:, 1]  # 不知道为什么，这里加一个负号就跟sklearn的PCA结果一样，否则不一样
plt.scatter(xVal, yVal)
plt.show()
