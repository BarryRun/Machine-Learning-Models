"""
两元素和为X。给定一个由n个实数构成的集合S 和另一个实数x，判断S中是否有两个元素的和为x。
试设计一个分治算法求解上述问题，并分析算法的时间复杂度。
"""


def divide_and_conquer(A, target):
    if len(A) <= 1:
        return False
    i = 0
    j = len(A) - 1

    while i < j:
        if A[i] + A[j] == target:
            return True
        elif A[i] + A[j] < target:
            i += 1
        else:
            j -= 1

    return False


if __name__ == '__main__':
    test_array = [1, 5, 7, 9, 4]
    print(divide_and_conquer(test_array, 15))