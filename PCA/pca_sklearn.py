from sklearn.decomposition import PCA
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

# 默认采用奇异值分解的方式，这里为了与我们自己写的pca做对比选择传统意义上的svd
pca = PCA(2, svd_solver='full')
data = pca.fit_transform(data)

# 作图查看降维效果
xVal = data[:, 0]
yVal = data[:, 1]
plt.scatter(xVal, yVal)
plt.show()

