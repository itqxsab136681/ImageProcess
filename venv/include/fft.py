# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:59:02 2019

@author: Chuck
"""

import numpy as np
import matplotlib.pyplot as plt


# 一维傅里叶实现图像二维傅里叶快速变换
def dft2D(f_temp):
    #    f_temp=plt.imread(f)
    row, col = f_temp.shape
    F = np.zeros([row, col], 'complex128')
    for u in range(row):
        F[u, :] = np.fft.fft(f_temp[u, :])
    for v in range(col):
        F[:, v] = np.fft.fft(F[:, v])
    return F


# 二维快速傅里叶逆变换
def idft2D(F):
    F = np.matrix(F)
    F_H = F.H
    f_MN = dft2D(F_H)
    f = f_MN / f_MN.size
    f = np.matrix(f)
    f = np.array(f.H)
    return f


# 合成矩形物体图像
def creat_square(s_size):
    f = np.zeros(s_size)
    for i in range(225, 285):
        for j in range(250, 260):
            f[i, j] = 1
    return f


def center_fft(F):
    row, col = F.shape
    temp_F = np.zeros([row, col], 'complex128')
    for i in range(255):
        for j in range(255):
            temp_F[255 + i, 255 + j] = F[i, j]
    for i in range(255):
        for j in range(255, 512):
            temp_F[255 + i, j - 255] = F[i, j]
    for i in range(255, 512):
        for j in range(255):
            temp_F[i - 255, j + 255] = F[i, j]
    for i in range(255, 512):
        for j in range(255, 512):
            temp_F[i - 255, j - 255] = F[i, j]
    return temp_F


# 定义main函数运行第三题
def main():
    f = plt.imread('rose512.tif')
    f = (f - min(f.reshape(f.size, 1))) / (max(f.reshape(f.size, 1)) - min(f.reshape(f.size, 1)))
    F = dft2D(f)
    #    F_1=np.fft.fft2(f)
    g = idft2D(F)
    #    g_1=np.fft.ifft2(F)
    #    g_2=np.fft.ifft2(F_1)
    plt.imshow(np.abs(f - g), 'gray')
    plt.title('comparison_between_ifft_and_sourcepic')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.savefig('comparison_between_ifft_and_sourcepic.tif')
    plt.show()
    f_create = creat_square([512, 512])
    plt.imshow(np.abs(f_create), 'gray')
    plt.title('original_pic')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.show()
    F_create = dft2D(f_create)
    F_center = center_fft(F_create)
    S = np.log10(1 + np.abs(F_center))
    plt.imshow(np.abs(F_create), 'gray')
    plt.title('pic_after_fft')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.savefig('the_create_square_after_fft.tif')
    plt.show()
    plt.imshow(np.abs(F_center), 'gray')
    plt.title('fft_center_transform')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.savefig('the_fft_after_center_transform.tif')
    plt.show()
    plt.imshow(S, 'gray')
    plt.title('fft_transform_and_log')
    plt.xlabel('U')
    plt.ylabel('V')
    plt.savefig('the_fft_after_center_transform.tif')
    plt.show()


if __name__ == '__main__':
    main()
