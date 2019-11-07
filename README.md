
## 机器学习算法实现
####K聚类

####感知器

####EM算法

![EM算法的过程](https://github.com/BarryRun/Algorithms-homework/blob/master/Machine%20Learning/em_res/em_process.gif)
***

# 简单的算法题
## 分治算法


第一题：

设X[0:n-1]和Y[0:n-1]为两个数组，每个数组中含有n个已排好序的数。试设计一个O(logn)时间的分治算法，找出X和Y的2n个数的中位数，并证明算法的时间复杂性为O(logn)

第二题：

有一实数序列𝑎_1,𝑎_2,…,𝑎_𝑁，若𝑖<𝑗 且 𝑎_𝑖>𝑎_𝑗，则(𝑎_𝑖,𝑎_𝑗)构成了一个逆序对，请使用分治方法求整个序列中逆序对个数，并分析算法的时间复杂性。

第三题：
给定𝑛座建筑物 𝐵[1, 2, … , 𝑛]，每个建筑物 𝐵[𝑖]表示为一个矩形，用三元组𝐵[𝑖]=(𝑎_𝑖,𝑏_𝑖,ℎ_𝑖)表示，其中𝑎_𝑖表示建筑左下顶点，𝑏_𝑖表示建筑的右下顶点，ℎ_𝑖表示建筑的高，请设计一个 𝑂(𝑛log𝑛)的算法求出这𝑛座建筑物的天际轮廓。例如，左下图所示中8座建筑的表示分别为(1,5,11), (2,7,6), (3,9,13), (12,16,7), (14,25,3), (19,22,18), (23,29,13)和(24,28,4)，其天际轮廓如右下图所示可用9个高度的变化(1, 11), (3, 13), (9, 0), (12, 7), (16, 3), (19, 18), (22, 3), (23, 13)和(29,0)表示。另举一个例子，假定只有一个建筑物(1, 5, 11)，其天际轮廓输出为2个高度的变化(1, 11), (5, 0)。

第四题：
最大子数组问题。一个包含n个整数（有正有负）的数组A，设计一O(nlogn)算法找出和最大的非空连续子数组。对于此问题你还能设计出O(n)的算法吗？

第五题：
循环移位问题。给定一个数组，数组中元素按从小到大排好序，现将数组中元素循环右移若干位，请设计一算法，计算出循环右移了多少位。

第六题：
两元素和为X。给定一个由n 个实数构成的集合S 和另一个实数x，判断S 中是否有两个元素的和为x。试设计一个分治算法求解上述问题，并分析算法的时间复杂度。


