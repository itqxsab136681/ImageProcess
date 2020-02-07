"""
Created on Sep 26 22:20:41 2019

@author: 秦小山
"""

import numpy as np

# 程序输入
n = int(input("请输入矩阵的阶数n：\n"))
A = []
print("请输入 n * n个矩阵元素(enter键间隔)：")
for i in range(0, n):
    A.append([])
    for j in range(0, n):
        A[i].append(float(input()))
A = np.mat(A)


# lu分解
def my_LU(B):
    A = np.array(B)
    n = len(A)

    L = np.zeros(shape=(n, n))
    U = np.zeros(shape=(n, n))

    for k in range(n - 1):

        gv = A[:, k]
        # 模拟高斯消去直接讲每一列一次置0
        gv[k + 1:] = gv[k + 1:] / gv[k]
        gv[0:k + 1] = np.zeros(k + 1)

        # l主对角线直接=1，非对角线一次填入gv
        L[:, k] = gv
        L[k][k] = 1.0
        for l in range(k + 1, n):
            B[l, :] = B[l, :] - gv[l] * B[k, :]

        A = np.array(B)
    L[k + 1][k + 1] = 1.0
    U = A
    print(U)
    print(L)


my_LU(A)
