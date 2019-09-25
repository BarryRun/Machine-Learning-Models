"""
有一实数序列𝑎_1,𝑎_2,…,𝑎_𝑁，若𝑖<𝑗 且 𝑎_𝑖>𝑎_𝑗，则(𝑎_𝑖,𝑎_𝑗)构成了一个逆序对，请使用分治方法求整个序列中逆序对个数，并分析算法的时间复杂性。
"""


def divide_and_conquer(array):
    n = len(array)
    if n == 1:
        return 0, array
    elif n == 2:
        if array[0] > array[1]:
            return 1, array[::-1]
        else:
            return 0, array

    mid = n//2
    left_num, left_array = divide_and_conquer(array[:mid+1])
    right_num, right_array = divide_and_conquer(array[mid+1:])

    sum_num = left_num + right_num
    i = j = 0
    new_array = []
    while True:
        if i == len(left_array):
            new_array = new_array + right_array[j:]
            break
        elif j == len(right_array):
            new_array = new_array + left_array[i:]
            break

        if left_array[i] <= right_array[j]:
            new_array.append(left_array[i])
            i += 1
        else:
            new_array.append(right_array[j])
            j += 1
            sum_num += (len(left_array) - i)

    return sum_num, new_array


if __name__ == '__main__':
    test_array = [1, 6, 8, 3, 1, 4, 5]
    res, res_array= divide_and_conquer(test_array)
    print(res, str(res_array))

