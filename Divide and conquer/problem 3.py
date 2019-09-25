"""
给定𝑛座建筑物 𝐵[1, 2, … , 𝑛]，每个建筑物 𝐵[𝑖]表示为一个矩形，用三元组𝐵[𝑖]=(𝑎_𝑖,𝑏_𝑖,ℎ_𝑖)表示
其中𝑎_𝑖表示建筑左下顶点，𝑏_𝑖表示建筑的右下顶点，ℎ_𝑖表示建筑的高
请设计一个 𝑂(𝑛log𝑛)的算法求出这𝑛座建筑物的天际轮廓
例如，8座建筑的表示分别为(1,5,11), (2,7,6), (3,9,13), (12,16,7), (14,25,3), (19,22,18), (23,29,13)和(24,28,4)
其天际轮廓可用9个高度的变化(1, 11), (3, 13), (9, 0), (12, 7), (16, 3), (19, 18), (22, 3), (23, 13)和(29,0)表示。
另举一个例子，假定只有一个建筑物(1, 5, 11)，其天际轮廓输出为2个高度的变化(1, 11), (5, 0)。
"""


# 该代码来自于leetcode， https://leetcode-cn.com/problems/the-skyline-problem/solution/tian-ji-xian-wen-ti-by-leetcode/
def getSkyline(buildings):
    """
    Divide-and-conquer algorithm to solve skyline problem,
    which is similar with the merge sort algorithm.
    """
    n = len(buildings)
    # The base cases
    if n == 0:
        return []
    if n == 1:
        x_start, x_end, y = buildings[0]
        return [[x_start, y], [x_end, 0]]

        # If there is more than one building,
    # recursively divide the input into two subproblems.
    left_skyline = getSkyline(buildings[: n // 2])
    right_skyline = getSkyline(buildings[n // 2:])

    # Merge the results of subproblem together.
    return merge_skylines(left_skyline, right_skyline)


def merge_skylines(left, right):
    """
    Merge two skylines together.
    """

    def update_output(x, y):
        """
        Update the final output with the new element.
        """
        # if skyline change is not vertical -
        # add the new point
        if not output or output[-1][0] != x:
            output.append([x, y])
        # if skyline change is vertical -
        # update the last point
        else:
            output[-1][1] = y

    def append_skyline(p, lst, n, y, curr_y):
        """
        Append the rest of the skyline elements with indice (p, n)
        to the final output.
        """
        while p < n:
            x, y = lst[p]
            p += 1
            if curr_y != y:
                update_output(x, y)
                curr_y = y

    n_l, n_r = len(left), len(right)
    p_l = p_r = 0
    curr_y = left_y = right_y = 0
    output = []

    # while we're in the region where both skylines are present
    while p_l < n_l and p_r < n_r:
        point_l, point_r = left[p_l], right[p_r]
        # pick up the smallest x
        if point_l[0] < point_r[0]:
            x, left_y = point_l
            p_l += 1
        else:
            x, right_y = point_r
            p_r += 1
        # max height (i.e. y) between both skylines
        max_y = max(left_y, right_y)
        # if there is a skyline change
        if curr_y != max_y:
            update_output(x, max_y)
            curr_y = max_y

    # there is only left skyline
    append_skyline(p_l, left, n_l, left_y, curr_y)

    # there is only right skyline
    append_skyline(p_r, right, n_r, right_y, curr_y)

    return output


if __name__ == '__main__':
    test_input= [ [2, 9, 10], [3, 7, 15], [5, 12, 12], [15, 20, 10], [19, 24, 8] ]
    print(str(getSkyline(test_input)))