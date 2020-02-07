import math

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def canny(f):
    blur = cv.GaussianBlur(f, (5, 5), 0)
    rows, cols = blur.shape
    contour = np.zeros(blur.shape)
    angle = np.zeros(blur.shape)

    maskX = np.array([-1, 1, -1, 1]).reshape(2, 2)
    maskY = np.array([1, 1, -1, -1]).reshape(2, 2)
    for i in range(0, rows - 1):
        for j in range(0, cols - 1):
            x = 0.5 * np.sum(blur[i:i + 2, j:j + 2] * maskX)
            y = 0.5 * np.sum(blur[i:i + 2, j:j + 2] * maskY)
            # 转换角度的时候应该取abs 转换到第一象限减少后面的比较正角度
            if x != 0 or y != 0:
                contour[i, j] = pow(pow(x, 2) + pow(y, 2), 0.5)
                angle[i, j] = math.atan2(y, x) / math.pi * 180

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if contour[i, j] != 0:
                if -22.5 < angle[i, j] < 22.5 or -157.5 < angle[i, j] < 157.5:
                    contour[i, j] = 0 if contour[i, j] != np.max(contour[i, j - 1:j + 2]) else contour[i, j]
                elif 22.5 <= angle[i, j] < 67.5 or -157.5 <= angle[i, j] < -112.5:
                    contour[i, j] = 0 if contour[i, j] != max(contour[i - 1, j + 1], contour[i, j],
                                                              contour[i + 1, j - 1]) else contour[i, j]
                elif 67.5 <= angle[i, j] < 112.5 or -112.5 <= angle[i, j] < -67.5:
                    contour[i, j] = 0 if contour[i, j] != np.max(contour[i - 1:i + 2, j]) else contour[i, j]
                else:
                    contour[i, j] = 0 if contour[i, j] != max(contour[i - 1, j - 1], contour[i, j],
                                                              contour[i + 1, j + 1]) else contour[i, j]
    m = np.max(contour)
    th1 = np.zeros(contour.shape)
    th2 = np.zeros(contour.shape)
    stack = []

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if contour[i, j] >= m * 0.8:
                th2[i, j] = 1
                th1[i, j] = 1
                stack.append((i, j))
            elif contour[i, j] >= m * 0.2:
                th1[i, j] = 1

    while True:
        temp = th2.copy()
        for i in range(1, rows - 1, 3):
            for j in range(1, cols - 1, 3):
                if th2[i, j] == 1:
                    th2[i - 1:i + 1, j - 1, j + 1] += + th1[i - 1:i + 1, j - 1, j + 1]
        if temp == th2:
            break

    # 判定依据有多种。有的人是判定弱边缘点的8邻域中是否存在强边缘，如果有则将弱边缘设置成强的。没有就认为是假边缘。
    #
    # 另一种方案是用搜索算法，通过强边缘点，搜索8领域是否存在弱边缘，如果有，以弱边缘点为中心继续搜索，直到搜索不到弱边缘截止。

    # while True:
    #     i, j = stack.pop()
    #     if th1[i - 1, j - 1] != 0:
    #         th2[i - 1, j - 1] = 1
    #         stack.append((i - 1, j - 1))
    #     if th1[i - 1, j] != 0:
    #         th2[i - 1, j] = 1
    #         stack.append((i - 1, j))
    #     if th1[i - 1, j + 1] != 0:
    #         th2[i - 1, j + 1] = 1
    #         stack.append((i - 1, j + 1))
    #     if th1[i, j - 1] != 0:
    #         th2[i, j - 1] = 1
    #         stack.append((i, j - 1))
    #     if th1[i, j + 1] != 0:
    #         th2[i, j + 1] = 1
    #         stack.append((i, j + 1))
    #     if th1[i + 1, j - 1] != 0:
    #         th2[i + 1, j - 1] = 1
    #         stack.append((i + 1, j - 1))
    #     if th1[i + 1, j] != 0:
    #         th2[i + 1, j] = 1
    #         stack.append((i + 1, j))
    #     if th1[i + 1, j + 1] != 0:
    #         th2[i + 1, j + 1] = 1
    #         stack.append((i + 1, j + 1))

    # if len(stack) == 0:
    #     break

    plt.figure()
    f_canny = cv.Canny(cv.GaussianBlur(f, (5, 5), 0), 50, 150)
    show(th2 * 255, "th2", 1, 2, 1)
    show(f_canny, "f_canny", 1, 2, 2)
    plt.show()

    return th2


def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


def main():
    f = plt.imread("cameraman.tif")
    canny(f)
    # plt.figure()
    # show(canny(f), "f_gray", 1, 2, 1)
    # show(f_canny, "f_canny", 1, 2, 2)
    # plt.show()


if __name__ == '__main__':
    main()
