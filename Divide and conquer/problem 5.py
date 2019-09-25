"""
循环移位问题。给定一个数组，数组中元素按从小到大排好序，现将数组中元素循环右移若干位，请设计一算法，计算出循环右移了多少位。
"""


# 假设不考虑重复元素
def divide_and_conquer(array, i, j):
    n = j - i + 1
    if n == 2:
        return i + 1
    if array[i] < array[j]:
        return 0

    mid = (i+j)//2
    if array[0] > array[mid]:
        return divide_and_conquer(array, i, mid)
    else:
        return divide_and_conquer(array, mid, j)


if __name__ == '__main__':
    test_array = [4, 11, 12, 13, 2, 3]
    print(divide_and_conquer(test_array, 0, len(test_array)-1))