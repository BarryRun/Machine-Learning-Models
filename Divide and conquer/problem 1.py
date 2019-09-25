"""
问题描述：
设X[0:n-1]和Y[0:n-1]为两个数组，每个数组中含有n个已排好序的数。试设计一个O(logn)时间的分治算法，找出X和Y的2n个数的中位数，并证明算法的时间复杂性为O(logn)
"""
import math


def divide_and_conquer(x, y):
    n = len(x)
    if n == 1:
        return (x[0] + y[0])/2.0
    if n == 2:
        if x[0] > y[0] and x[1] < y[1]:
            return (x[0] + x[1])/2
        if x[0] < y[0] and x[1] > y[1]:
            return (y[0] + y[1])/2

    if n % 2 == 0:
        x_mid = (x[int(n/2)] + x[int((n-2)/2)]) / 2.0
        y_mid = (y[int(n/2)] + y[int((n-2)/2)]) / 2.0
    else:
        x_mid = x[int((n-1)/2)]
        y_mid = y[int((n-1)/2)]

    if x_mid < y_mid:
        return divide_and_conquer(x[int((n-1)/2):], y[:math.ceil((n+1)/2)])
    elif x_mid > y_mid:
        return divide_and_conquer(x[:math.ceil((n+1)/2)], y[int((n-1)/2):])
    else:
        return x_mid


if __name__ == '__main__':
    y = [1, 50, 60, 61]
    x = [-100, -50, 160, 999]
    if len(x) != len(y) or len(x) == 0:
        print("输入有误！")
    else:
        print(divide_and_conquer(x, y))


