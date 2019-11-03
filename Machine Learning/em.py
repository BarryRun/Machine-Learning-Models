import numpy as np
import matplotlib.pyplot as plt
import math


# 高斯分布的概率密度函数
def normal_distribution(x, mu, sigma):
    return np.exp(-1*((x-mu)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


# 生成一个由三个一维高斯分布加权生成的一维数据
def data_generation(mu1, sigma1, alpha1, mu2, sigma2, alpha2, mu3, sigma3, alpha3):
    first_guass = np.random.normal(mu1, sigma1, 500)
    second_guass = np.random.normal(mu2, sigma2, 500)
    third_guass = np.random.normal(mu3, sigma3, 500)
    data = alpha1 * first_guass + alpha2 * second_guass + alpha3 * third_guass
    print(len(data))
    print(data)

    # 绘制三个高斯分布的概率密度曲线
    x1 = np.linspace(mu1 - 6 * sigma1, mu1 + 6 * sigma1, 100)
    x2 = np.linspace(mu2 - 6 * sigma2, mu2 + 6 * sigma2, 100)
    x3 = np.linspace(mu3 - 6 * sigma3, mu3 + 6 * sigma3, 100)

    y1 = normal_distribution(x1, mu1, sigma1)
    y2 = normal_distribution(x2, mu2, sigma2)
    y3 = normal_distribution(x3, mu3, sigma3)

    plt.plot(x1, y1, 'r', label='m='+str(mu1)+',sig='+str(sigma1))
    plt.plot(x2, y2, 'g', label='m='+str(mu2)+',sig='+str(sigma2))
    plt.plot(x3, y3, 'b', label='m='+str(mu3)+',sig='+str(sigma3))
    # 同时绘制出数据分布的地方
    plt.scatter(data, [0.05]*500, s=1)
    plt.legend()
    plt.grid()
    plt.show()

    return data


if __name__ == '__main__':
    train_data = data_generation(2, 2.5, 0.4, -3, 1.5, 0.4, 5, 1.7, 0.2)
