import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import math as mt
from sklearn import preprocessing


# 傅立叶正变换 缺省参数为0，默认正向，因为正反傅立叶变换其实傅立叶变换的操作都是一致的所以加了个缺省参数，组教大人求放过啊
def dft2D(f, type=0):
    rows, cols = f.shape
    F = np.zeros([rows & -2, cols & -2], 'complex128')  # 与上-2取最小偶数加快傅立叶变换速度
    # 翻变换时图像先取共轭
    if type == 1:
        f = np.matrix(f).H

    for u in range(rows):
        F[u, :] = np.fft.fft(f[u, :])
    for v in range(cols):
        F[:, v] = np.fft.fft(F[:, v])

    # 傅立叶逆变换 傅立叶后再经过正交变换得到原图像
    if type == 1:
        F = F / F.size
        F = np.array(np.matrix(F).H)

    return F


# 傅立叶反变换
def idft2D(F):
    return dft2D(F, 1)


# 图像显示函数抽取
def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


def main():
    f = plt.imread("dd.jpg")
    # f = cv.imread("original.tif")
    r, g, b = cv.split(f)
    f_d_grayr = hp(r)
    f_d_grayg = hp(g)
    f_d_grayb = hp(b)

    newf = cv.merge([f_d_grayr, f_d_grayg, f_d_grayb])

    plt.figure()
    show(f, "f", 1, 2, 1)
    show(newf, "cc", 1, 2, 2)
    plt.show()

    # F = np.abs(np.abs(F) - f)
    # F.astype(np.uint32)
    # cv.imshow("cc", F)
    # cv.waitKey(0)

    plt.figure()
    F = np.zeros([512, 512])
    F[226:287, 251:262] = 255
    show(F, 'matrix', 2, 2, 1)

    # 生成矩阵的傅立叶变换
    F = F / 255.0
    F = dft2D(F, 0)
    show(np.abs(F), 'DFT', 2, 2, 2)

    F = np.zeros([512, 512])
    F[226:287, 251:262] = 255
    F = F / 255.0
    # 中心化预处理
    for i in range(F.shape[0]):
        for j in range(F.shape[1]):
            F[i, j] = F[i, j] * ((-1) ** (i + j))
    # 中心化的傅立叶变换展示
    F = dft2D(F, 0)
    show(np.abs(F), 'DFT_center', 2, 2, 3)

    # 对数化的傅立叶变换中心化
    F = np.log(1 + np.abs(F))
    show(np.abs(F), 'DFT_center_log', 2, 2, 4)
    plt.show()


def hp(f):
    rows, cols = f.shape
    # 对数化区分fi fx numpy的溢出若+1则 255的时候因为是uint 数据上溢 最后1 0000 0000 1被截断所以值为0 log 后为负无穷
    # >> > np.log([1, np.e, np.e ** 2, 0])
    # array([0., 1., 2., -Inf])
    # pyhton 中的溢出，短整形会自动调整为长整型
    f_gray_log = np.log(1e-3 + f)
    f_d = np.fft.fftshift(np.fft.fft2(f_gray_log))
    f_d_mask = np.zeros(f_gray_log.shape)
    DX = max(rows, cols)
    for i in range(rows):
        for j in range(cols):
            temp = (i - rows / 2) ** 2 + (j - cols / 2) ** 2
            f_d_mask[i, j] = (2 - 0.2) * (1 - np.exp(- 0.1 * temp / 2 * (DX ** 2))) + 0.2
    f_d1 = np.fft.ifftshift(f_d * f_d_mask)

    f_d_gray1 = np.fft.ifft2(f_d1).real
    f_d_gray1 = np.exp(f_d_gray1) - 1
    mi = np.min(f_d_gray1)
    ma = np.max(f_d_gray1)
    rang = ma - mi
    for i in range(rows):
        for j in range(cols):
            f_d_gray1[i, j] = (f_d_gray1[i, j] - mi) / rang
    return f_d_gray1


if __name__ == '__main__':
    main()
