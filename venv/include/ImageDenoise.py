import cv2
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


# 高斯噪声函数单行
def wgn(x, snr):
    snr = 10 ** (snr / 10.0)
    xpower = np.sum(x ** 2) / len(x)
    npower = xpower / snr
    return np.random.randn(len(x)) * np.sqrt(npower)


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
            maskList = []
            array_var = []
            maskList.append(noise[i - 2:i + 1, j - 2:j + 1] * maskHexagon1)
            maskList.append(noise[i - 2:i + 1, j - 1:j + 2] * maskPentTop)
            maskList.append(noise[i - 2:i + 1, j:j + 3] * maskHexagon2)
            maskList.append(noise[i - 1:i + 2, j - 2:j + 1] * maskPentLeft)
            maskList.append(noise[i - 1:i + 2, j - 1:j + 2] * maskRect)
            maskList.append(noise[i - 1:i + 2, j:j + 3] * maskPentRight)
            maskList.append(noise[i:i + 3, j - 2:j + 1] * maskHexagon3)
            maskList.append(noise[i:i + 3, j - 1:j + 2] * maskPentBottom)
            maskList.append(noise[i:i + 3, j:j + 3] * maskHexagon4)

            array_var.append(np.var(noise[i - 2:i + 1, j - 2:j + 1] * maskHexagon1))
            array_var.append(np.var(noise[i - 2:i + 1, j - 1:j + 2] * maskPentTop))
            array_var.append(np.var(noise[i - 2:i + 1, j:j + 3] * maskHexagon2))
            array_var.append(np.var(noise[i - 1:i + 2, j - 2:j + 1] * maskPentLeft))
            array_var.append(np.var(noise[i - 1:i + 2, j - 1:j + 2] * maskRect))
            array_var.append(np.var(noise[i - 1:i + 2, j:j + 3] * maskPentRight))
            array_var.append(np.var(noise[i:i + 3, j - 2:j + 1] * maskHexagon3))
            array_var.append(np.var(noise[i:i + 3, j - 1:j + 2] * maskPentBottom))
            array_var.append(np.var(noise[i:i + 3, j:j + 3] * maskHexagon4))

            deNoise[i, j] = np.mean(maskList[array_var.index(min(array_var))])

    return deNoise


def main():
    original = plt.imread("lena.tiff", 0)
    rows, cols = original.shape
    original_noise = original.copy().astype(np.float64)

    # 生成噪声图像，信噪比为10
    for i in range(cols):
        original_noise[:, i] += wgn(original_noise[:, i], 10)

    ImageDenoise = edgeFilter(original_noise)

    plt.figure()
    show(original, "original", 2, 2, 1)
    show(original_noise, "original_noise", 2, 2, 2)
    show(ImageDenoise, "ImageDenoise", 2, 2, 3)
    show(original - ImageDenoise, "original - ImageDenoise", 2, 2, 4)
    plt.show()

    mask = np.ones(9).reshape(3, 3)
    ImageDenoise = meanDenoise(mask, original)
    plt.figure()
    show(original, "original", 2, 2, 1)
    show(original_noise, "original_noise", 2, 2, 2)
    show(ImageDenoise, "ImageDenoise", 2, 2, 3)
    show(original - ImageDenoise, "original - ImageDenoise", 2, 2, 4)
    plt.show()


def meanDenoise(mask, original):
    rows, cols = original.shape
    ImageDenoise = np.zeros(original.shape)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            ImageDenoise[i, j] = np.mean(original[i - 1:i + 2, j - 1:j + 2] * mask)
    return ImageDenoise


if __name__ == '__main__':
    main()
