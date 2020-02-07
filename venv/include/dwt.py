import random

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pywt as pt


def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


def main():
    original = plt.imread("lena.tiff")
    rows, cols = original.shape
    original_b = original.copy().astype(np.float64)
    for i in range(cols):
        original_b[:, i] += wgn(original_b[:, i], 10)
        # 生成噪声图像

    # point = []
    # for i in range(int(rows * cols / 20)):
    #     point.append([random.randint(0, 511), random.randint(0, 511)])
    # for it in point:
    #     x, y = it
    #     original_b[x, y] = 0

    [cA3, (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pt.wavedec2(original_b, 'bior4.4', mode='symmetric',
                                                                           level=3, axes=(-2, -1))

    d1 = cD1.reshape(1, -1)
    d2 = cD2.reshape(1, -1)
    d3 = cD3.reshape(1, -1)

    d = np.zeros(d1.shape[1] + d2.shape[1] + d3.shape[1])
    d[0:d1.shape[1]] = d1
    d[d1.shape[1]:d1.shape[1] + d2.shape[1]] = d2
    d[d1.shape[1] + d2.shape[1]:d1.shape[1] + d2.shape[1] + d3.shape[1]] = d3

    xn = np.median(np.abs(d)) / 0.6745
    cd = [cD1, cD2, cD3, cH1, cH2, cH3, cV1, cV2, cV3]
    for k, d in enumerate(cd):
        rowsH, colsH = d.shape
        y = np.sum(d * d) / (rowsH * colsH)
        x = y - xn ** 2

        mul = x / (x + xn ** 2)
        d *= mul

    ca2 = pt.idwt2((cA3, (cH3, cV3, cD3)), 'bior4.4')
    ca1 = pt.idwt2((ca2, (cH2, cV2, cD2)), 'bior4.4')
    newI = pt.idwt2((ca1, (cH1, cV1, cD1)), 'bior4.4')
    plt.figure()
    show(original, "original", 2, 2, 1)
    show(original_b, "original_b", 2, 2, 2)
    show(newI, "newI", 2, 2, 3)
    show(original_b - newI, "-", 2, 2, 4)
    plt.show()


if __name__ == '__main__':
    main()
