import math
import random
import time
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def myEqualizeHist(I):
    rows, cols = I.shape
    array = np.zeros(256)

    # 统计像素个数
    for i in range(rows):
        for j in range(cols):
            array[I[i][j]] += 1

    # 算累计概率
    sum = 0
    for ary in range(0, len(array)):
        sum += array[ary] / (rows * cols)
        array[ary] = sum

    # 求均衡化后的像素值
    newI = np.zeros(I.shape)
    for i in range(rows):
        for j in range(cols):
            newI[i][j] = array[I[i][j]] * 255

    return newI


# 添加椒盐噪声
def pepperNoise(count, I):
    rows, cols = I.shape
    newI = np.copy(I)
    for i in range(count):
        if i % 2 == 0:
            newI[random.randint(0, rows - 1)][random.randint(0, cols - 1)] = 0
        else:
            newI[random.randint(0, rows - 1)][random.randint(0, cols - 1)] = 255
    return newI


# 有选择保边缘平滑方法
def edgeFilter(noise):
    rows, cols = noise.shape
    deNoise = np.zeros(noise.shape)
    # 生成滤波模板下标数组
    maskRect = np.ones(9).reshape(3, 3)

    maskPentTop = np.array([1, 1, 1, 1, 1, 1, 0, 1, 0]).reshape(3, 3)
    maskPentLeft = np.array([1, 1, 0, 1, 1, 1, 1, 1, 0]).reshape(3, 3)
    maskPentBottom = np.array([0, 1, 0, 1, 1, 1, 1, 1, 1]).reshape(3, 3)
    maskPentRight = np.array([0, 1, 1, 1, 1, 1, 0, 1, 1]).reshape(3, 3)

    maskHexagon1 = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1]).reshape(3, 3)
    maskHexagon2 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0]).reshape(3, 3)
    maskHexagon3 = np.array([0, 1, 1, 1, 1, 1, 1, 1, 0]).reshape(3, 3)
    maskHexagon4 = np.array([1, 1, 0, 1, 1, 1, 0, 1, 1]).reshape(3, 3)

    for i in range(2, rows - 2):
        for j in range(2, cols - 2):
            array_mean = []
            array_var = []
            array_mean.append(np.mean(noise[i - 2:i + 1, j - 2:j + 1] * maskHexagon1))
            array_mean.append(np.mean(noise[i - 2:i + 1, j - 1:j + 2] * maskPentTop))
            array_mean.append(np.mean(noise[i - 2:i + 1, j:j + 3] * maskHexagon2))
            array_mean.append(np.mean(noise[i - 1:i + 2, j - 2:j + 1] * maskPentLeft))
            array_mean.append(np.mean(noise[i - 1:i + 2, j - 1:j + 2] * maskRect))
            array_mean.append(np.mean(noise[i - 1:i + 2, j:j + 3] * maskPentRight))
            array_mean.append(np.mean(noise[i:i + 3, j - 2:j + 1] * maskHexagon3))
            array_mean.append(np.mean(noise[i:i + 3, j - 1:j + 2] * maskPentBottom))
            array_mean.append(np.mean(noise[i:i + 3, j:j + 3] * maskHexagon4))

            array_var.append(np.var(noise[i - 2:i + 1, j - 2:j + 1] * maskHexagon1))
            array_var.append(np.var(noise[i - 2:i + 1, j - 1:j + 2] * maskPentTop))
            array_var.append(np.var(noise[i - 2:i + 1, j:j + 3] * maskHexagon2))
            array_var.append(np.var(noise[i - 1:i + 2, j - 2:j + 1] * maskPentLeft))
            array_var.append(np.var(noise[i - 1:i + 2, j - 1:j + 2] * maskRect))
            array_var.append(np.var(noise[i - 1:i + 2, j:j + 3] * maskPentRight))
            array_var.append(np.var(noise[i:i + 3, j - 2:j + 1] * maskHexagon3))
            array_var.append(np.var(noise[i:i + 3, j - 1:j + 2] * maskPentBottom))
            array_var.append(np.var(noise[i:i + 3, j:j + 3] * maskHexagon4))

            deNoise[i, j] = array_mean[array_var.index(min(array_var))]

    return deNoise[2:rows - 2, 2:cols - 2]


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
    original = plt.imread("cameraman.tif", 0)
    myE = myEqualizeHist(original)
    openE = cv.equalizeHist(original)
    plt.figure()
    show(original, "original", 2, 2, 1)
    show(myE, "myE", 2, 2, 2)
    show(openE, "openE", 2, 2, 3)
    c = openE - myE
    show(c, "openE - myE", 2, 2, 4)
    plt.show()

    start = time.time()
    # 图片保边缘平滑去噪
    noise = pepperNoise(512, original)
    plt.figure()
    show(noise, "noiseI", 1, 2, 1)
    newI = edgeFilter(noise)
    show(newI, "deblur", 1, 2, 2)
    print(time.time() - start)
    plt.show()

    # 拉普拉斯增强
    bf = cv.GaussianBlur(original, (5, 5), 0)
    enImage = laplacianEnhance(bf)
    plt.figure()
    show(original, "original", 2, 2, 1)

    ret, dst1 = cv.threshold(original + enImage, 255, 255, cv.THRESH_TRUNC)
    show(dst1, "enImage", 2, 2, 2)
    ret, dst2 = cv.threshold(original + cv.Laplacian(bf, ddepth=-1), 255, 255, cv.THRESH_TRUNC)
    show(dst2, "cv.Laplacian(f)", 2, 2, 3)

    show(dst1 - dst2, "-", 2, 2, 4)

    plt.show()


if __name__ == '__main__':
    main()
