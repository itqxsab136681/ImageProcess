import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import random
import math


def histequal4e(I):
    rows, cols = I.shape
    array = np.zeros(256)

    for i in range(rows):
        for j in range(cols):
            array[I[i][j]] += 1

    sum = 0
    for ary in range(0, len(array)):
        sum += array[ary] / (rows * cols)
        array[ary] = sum

    newI = np.zeros(I.shape)
    for i in range(rows):
        for j in range(cols):
            newI[i][j] = array[I[i][j]] * 255

    return newI


def pepperNoise(count, I):
    rows, cols = I.shape
    newI = np.copy(I)
    for i in range(count):
        if i % 2 == 0:
            newI[random.randint(0, rows - 1)][random.randint(0, cols - 1)] = 0
        else:
            newI[random.randint(0, rows - 1)][random.randint(0, cols - 1)] = 255
    return newI


def edgeFilter(noiseI):
    rows, cols = noiseI.shape
    # 生成滤波模板下标数组
    maskRect = np.array([1 + cols, 2 + cols, 3 + cols, 1 + cols * 2, 2 + cols * 2, 3 + cols * 2, 1 + cols * 3,
                         2 + cols * 3, 3 + cols * 3])

    maskPent1 = np.array([1, 2, 3, 1 + cols, 2 + cols, 3 + cols, 2 + cols * 2])
    maskPent2 = np.array([cols, 1 + cols, cols * 2, cols * 2 + 1, cols * 2 + 2, cols * 3, cols * 3 + 1])
    maskPent3 = np.array([cols * 2 + 2, cols * 3 + 1, cols * 3 + 2, cols * 3 + 3, cols * 4 + 1, cols * 4 + 2,
                          cols * 4 + 3])
    maskPent4 = np.array([cols * 1 + 3, cols * 1 + 4, cols * 2 + 2, cols * 2 + 3, cols * 2 + 4, cols * 3 + 3,
                          cols * 3 + 4])

    maskHexagon1 = np.array([0, 1, cols, cols + 1, cols + 2, cols * 2 + 1, cols * 2 + 2])
    maskHexagon2 = np.array([3, 4, cols + 2, cols + 3, cols + 4, cols * 2 + 2, cols * 2 + 3])
    maskHexagon3 = np.array([cols * 2 + 1, cols * 2 + 2, cols * 3, cols * 3 + 1, cols * 3 + 2, cols * 4, cols * 4 + 1])
    maskHexagon4 = np.array([cols * 2 + 2, cols * 2 + 3, cols * 3 + 2, cols * 3 + 3, cols * 3 + 4, cols * 4 + 3,
                             cols * 4 + 4])

    newI = np.zeros(noiseI.shape)
    array_mean = np.ndarray(9)
    array_var = np.ndarray(9)
    noiseI = noiseI.flatten()
    for i in range(rows):
        for j in range(cols):
            array_var = np.zeros(array_var.shape)
            array_mean = np.zeros(array_mean.shape)
            # 对边缘不作处理
            if 1 < i < rows - 4 and 1 < j < cols - 4:
                for m in range(9):
                    array_mean[0] += noiseI[maskRect[m] + i * cols + j] / 9
                    if m < 7:
                        array_mean[1] += noiseI[maskPent1[m] + i * cols + j] / 7
                        array_mean[2] += noiseI[maskPent2[m] + i * cols + j] / 7
                        array_mean[3] += noiseI[maskPent3[m] + i * cols + j] / 7
                        array_mean[4] += noiseI[maskPent4[m] + i * cols + j] / 7
                        array_mean[5] += noiseI[maskHexagon1[m] + i * cols + j] / 7
                        array_mean[6] += noiseI[maskHexagon2[m] + i * cols + j] / 7
                        array_mean[7] += noiseI[maskHexagon3[m] + i * cols + j] / 7
                        array_mean[8] += noiseI[maskHexagon4[m] + i * cols + j] / 7

                for n in range(9):
                    array_var[0] += math.pow(noiseI[maskRect[n] + i * cols + j] - array_mean[0], 2)
                    if n < 7:
                        array_var[1] += math.pow(noiseI[maskPent1[n] + i * cols + j] - array_mean[1], 2)
                        array_var[2] += math.pow(noiseI[maskPent2[n] + i * cols + j] - array_mean[2], 2)
                        array_var[3] += math.pow(noiseI[maskPent3[n] + i * cols + j] - array_mean[3], 2)
                        array_var[4] += math.pow(noiseI[maskPent4[n] + i * cols + j] - array_mean[4], 2)
                        array_var[5] += math.pow(noiseI[maskHexagon1[n] + i * cols + j] - array_mean[5], 2)
                        array_var[6] += math.pow(noiseI[maskHexagon2[n] + i * cols + j] - array_mean[6], 2)
                        array_var[7] += math.pow(noiseI[maskHexagon3[n] + i * cols + j] - array_mean[7], 2)
                        array_var[8] += math.pow(noiseI[maskHexagon4[n] + i * cols + j] - array_mean[8], 2)

                newI[i][j] = array_mean[np.argmin(array_var)]
            else:
                newI[i][j] = noiseI[i * cols + j]

    return newI


def laplacianEnhance(f):
    rows, cols = f.shape
    f = np.array(f)
    newI = np.zeros(f.shape, dtype=np.float)
    # opencv 里默认的3*3模板
    mask = np.array([-2, 0, -2, 0, 8, 0, -2, 0, -2]).reshape(3, 3)
    for i in range(rows):
        for j in range(cols):
            if 1 < i < rows - 2 and 1 < j < cols - 2:
                newI[i][j] = np.sum(np.dot(f[i - 1:i + 2, j - 1:j + 2], mask))
            else:
                newI[i][j] = f[i][j]

    newI = newI.clip(0, 255)
    return newI


def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


def main():
    # 直方图均衡
    f = plt.imread("cameraman.tif", 0)
    plt.figure()
    show(f, "original", 2, 2, 1)
    newf = histequal4e(f)
    show(newf, "changed", 2, 2, 2)
    equ = cv.equalizeHist(f)
    show(equ, "changequ", 2, 2, 3)
    c = equ - newf
    show(c, "c", 2, 2, 4)
    plt.show()

    # 图片平滑去噪
    noiseI = pepperNoise(5120, f)
    plt.figure()
    show(noiseI, "noiseI", 1, 2, 1)
    newI = edgeFilter(noiseI)
    show(newI, "deblur", 1, 2, 2)
    plt.show()

    # 拉普拉斯增强
    bf = cv.GaussianBlur(f, (5, 5), 0)
    enImage = laplacianEnhance(bf)
    plt.figure()
    show(f, "original", 2, 2, 1)

    ret, dst1 = cv.threshold(f + enImage, 255, 255, cv.THRESH_TRUNC)
    show(dst1, "enImage", 2, 2, 2)
    ret, dst2 = cv.threshold(f + cv.Laplacian(bf, ddepth=-1), 255, 255, cv.THRESH_TRUNC)
    show(dst2, "cv.Laplacian(f)", 2, 2, 3)

    show(dst1 - dst2, "-", 2, 2, 4)

    plt.show()


if __name__ == '__main__':
    main()
