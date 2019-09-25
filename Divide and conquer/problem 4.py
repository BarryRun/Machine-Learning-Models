"""
最大子数组问题。一个包含n个整数（有正有负）的数组A。
设计一O(nlogn)算法找出和最大的非空连续子数组。
对于此问题你还能设计出O(n)的算法吗？
"""


# 由于算法要求复杂度为O(nlogn)，因此考虑分治算法
def divide_and_conquer(array:list):
    n = len(array)
    if n == 1:
        return array[0]
    elif n == 2:
        return max(array[0], array[1], array[0]+array[1])

    mid = n//2
    left_max = divide_and_conquer(array[:mid+1])
    right_max = divide_and_conquer(array[mid+1:])

    i = mid
    j = mid + 1
    buf = sum_max = array[i] + array[j]
    while i != 0:
        i -= 1
        buf = buf + array[i]
        if buf > sum_max:
            sum_max = buf

    buf = sum_max
    while j != n - 1:
        j += 1
        buf = buf + array[j]
        if buf > sum_max:
            sum_max = buf

    return max(left_max, right_max, sum_max)


if __name__ == '__main__':
    test_array = [-1, -1, -1, -1, -1]
    print(divide_and_conquer(test_array))