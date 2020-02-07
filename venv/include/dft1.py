import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv


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
    f1 = plt.imread("cameraman.tif", 0)
    f = f1.copy()
    plt.figure()
    f3 = np.zeros(f.shape)
    show(f, "original", 2, 2, 1)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            f[i, j] = f[i, j] * ((-1) ** (i + j))
    show(f3, 'f', 2, 2, 2)
    F = dft2D(f)
    mask = np.zeros(f.shape)
    rows, cols = f.shape
    # for i in range(rows):
    #     for j in range(cols):
    #         temp = (i - (rows - 1 / 2 + 1.0)) ** 2 + (j - (cols - 1 / 2 + 1.0)) ** 2
    #         if temp < 200 ** 2:
    #             mask[i, j] = 0
    #         else:
    #             mask[i, j] = 1
    # F = F * mask
    F = idft2D(F)
    for i in range(f.shape[0]):
        for j in range(f.shape[1]):
            F[i, j] = F[i, j] * ((-1) ** (i + j))
    show(np.abs(F), 'def_changed', 2, 2, 3)
    show(np.abs(np.abs(F) - f1), 'f-g', 2, 2, 4)
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


if __name__ == '__main__':
    main()
