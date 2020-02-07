import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import sys


# 高帽运算
def topHat(f):
    rows, cols = f.shape
    newF = f.copy()
    newF1 = f.copy()
    for i in range(5, rows - 5):
        for j in range(5, cols - 5):
            if f[i, j] != 0:
                newF[i, j] = np.min(f[i - 3:i + 4, j - 3: j + 4])

    for i in range(5, rows - 5):
        for j in range(5, cols - 5):
            if f[i, j] != 0:
                newF1[i, j] = np.max(newF[i - 3:i + 4, j - 3: j + 4])

    return (f - newF1).clip(0, 255).astype("uint8")


# 腐蚀运算
def imerode(f, B):
    rowsB, colsB = B.shape
    rb = int(rowsB / 2)
    cb = int(colsB / 2)
    f = copyMakeBorder(f)
    newF = np.ndarray(f.shape)
    rows, cols = f.shape
    n = np.sum(B)
    for i in range(rb, rows - rb):
        for j in range(cb, cols - cb):
            newF[i, j] = 1 if np.sum(f[i - rb:i + rb + 1, j - cb:j + cb + 1] * B) == n else 0
    return newF[1:-1, 1:-1]


# 膨胀
def imdilate(f, B=np.ones(9).reshape(3, 3)):
    rowsB, colsB = B.shape
    rb = int(rowsB / 2)
    cb = int(colsB / 2)
    f = copyMakeBorder(f)
    newF = np.ndarray(f.shape)
    rows, cols = f.shape
    for i in range(rb, rows - rb):
        for j in range(cb, cols - cb):
            newF[i, j] = 1 if np.sum(f[i - rb:i + rb + 1, j - cb:j + cb + 1] * B) > 1 else 0
    return newF[1:-1, 1:-1]


# 开运算
def lockageon(f):
    f = f / 255
    mask1 = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape(3, 3)
    mask2 = np.ones(9).reshape(3, 3)
    return imdilate(imerode(f, mask1), mask2)


# 闭运算
def lockageoff(f):
    mask1 = np.array([0, 1, 0, 1, 1, 1, 0, 1, 0]).reshape(3, 3)
    mask2 = np.ones(9).reshape(3, 3)
    return imerode(imdilate(f, mask2), mask1)


# 击中击不中运算
def hitormiss(f, B, i, j):
    rows, cols = B.shape
    newB = f[(i - int((rows + 1) / 2)):i + int((rows + 1) / 2 + 1), j - int((cols + 1) / 2):j + int((cols + 1) / 2 + 1)]
    # newB[1:-1, 1:-1] = B
    if np.sum(newB[1:-1, 1:-1] * B) == np.sum(B):
        inversef = np.ones(f.shape) - f
        newB[1:-1, 1:-1] -= B
        return True if np.sum(imerode(f, B) * imerode(inversef, newB)) > 1 else False
    else:
        return False


def refining(f):
    rows, cols = f.shape
    # 细化模板
    B1 = np.array([-1, -1, -1, 0, 1, 0, 1, 1, 1]).reshape(3, 3)
    B2 = np.array([0, -1, -1, 1, 1, -1, 1, 1, 0]).reshape(3, 3)
    B3 = np.array([1, 0, -1, 1, 1, -1, 1, 0, -1]).reshape(3, 3)
    B4 = np.array([1, 1, 0, 1, 1, -1, 0, -1, -1]).reshape(3, 3)
    B5 = np.array([1, 1, 1, 0, 1, 0, -1, -1, -1]).reshape(3, 3)
    B6 = np.array([0, 1, 1, -1, 1, 1, -1, -1, 0]).reshape(3, 3)
    B7 = np.array([-1, 0, 1, -1, 1, 1, -1, 0, 1]).reshape(3, 3)
    B8 = np.array([-1, -1, 0, -1, 1, 1, 0, 1, 1]).reshape(3, 3)
    maskList = [B1, B2, B3, B4, B5, B6, B7, B8]
    count = 0
    skemask1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1]).reshape(3, 3)
    while True:
        temp = f.copy
        for m in maskList:
            mas = []
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if f[i, j] == 0:
                        continue
                    elif np.sum(m * f[i - 1:i + 2, j - 1:j + 2]) == 4:
                        # 击中时标记删除点
                        mas.append((i, j))
            for it in mas:
                x, y = it
                f[x, y] = 0
        if (temp == f).all:
            count += 1
        else:
            count = 0
        if count == 8:
            # 当8次没有一个击中的时候退出，退出时删除细化后剩余多余的点
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if f[i, j] == 1 and f[i - 1, j + 1] == 1 and f[i + 1, j - 1] == 1:
                        f[i - 1:i + 2, j - 1:j + 2] *= skemask1
                    if f[i, j] == 1 and f[i - 1, j - 1] == 1 and f[i + 1, j + 1] == 1:
                        f[i - 1:i + 2, j - 1:j + 2] *= skemask1
            break
    return f


# 裁剪算法
def cut(f):
    rows, cols = f.shape
    f2 = f.copy()
    # 裁剪模板
    A = np.array([0, -1, -1, 1, 1, -1, 0, -1, -1]).reshape(3, 3)
    A1 = np.rot90(A)
    A2 = np.rot90(A1)
    A3 = np.rot90(A2)
    B = np.array([1, -1, -1, -1, 1, -1, -1, -1, -1]).reshape(3, 3)
    B1 = np.rot90(B)
    B2 = np.rot90(B1)
    B3 = np.rot90(B2)
    maskList = [A, A1, A2, A3, B, B1, B2, B3]
    for k in range(3):
        for m in maskList:
            mas = []
            for i in range(1, rows - 1):
                for j in range(1, cols - 1):
                    if f2[i, j] == 0:
                        continue
                    elif np.sum(m * f2[i - 1:i + 2, j - 1:j + 2]) == 2:
                        mas.append((i, j))
            for it in mas:
                x, y = it
                f2[x, y] = 0

    # 得到首位端点，即膨胀初始点
    f3 = np.zeros(f.shape)
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if f2[i, j] == 0:
                continue
            else:
                for m in maskList:
                    if np.sum(m * f2[i - 1:i + 2, j - 1:j + 2]) == 2:
                        f3[i, j] = 1
    # 膨胀
    H = np.ones(9).reshape(3, 3)
    rows, cols = f3.shape
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            # 每一次膨胀与上A
            f3[i, j] = f[i, j] * (1 if np.sum(f3[i - 1:i + 2, j - 1:j + 2] * H) >= 1 else 0)

    newf = (f2 + f3) / 2
    ret, newf = cv.threshold(newf, 0, 1, cv.THRESH_BINARY)
    return newf


# 距离变换骨架提取
def distance(f):
    f2 = np.zeros(f.shape)
    rows, cols = f.shape

    # 距离变换操作
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            f[i, j] = sys.maxsize if f[i, j] == 0 else 1

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            temp0 = f[i, j]
            temp1 = min(f[i, j - 1] + 3, temp0)
            temp2 = min(f[i - 1, j - 1] + 4, temp1)
            temp3 = min(f[i - 1, j] + 3, temp2)
            temp4 = min(f[i - 1, j + 1] + 4, temp3)
            f[i, j] = temp4

    for i in range(rows - 2, 0, -1):
        for j in range(cols - 2, 0, -1):
            temp0 = f[i, j]
            temp1 = min(f[i, j + 1] + 3, temp0)
            temp2 = min(f[i + 1, j + 1] + 4, temp1)
            temp3 = min(f[i + 1, j] + 3, temp2)
            temp4 = min(f[i + 1, j - 1] + 4, temp3)
            f[i, j] = temp4

    # 骨架提取
    for i in range(3, rows - 3):
        for j in range(3, cols - 3):
            if 5 <= f[i, j] <= 10:
                if f[i, j] == np.max(f[i - 3:i + 4, j - 3:j + 4]):
                    f2[i, j] = 1
                elif np.sum(f[i - 1, j - 1:j + 2] * [-1, -1, 1]) == 1:
                    f2[i, j] = 1
                elif np.sum(f[i - 1, j - 1:j + 2] * [-1, -1, 1]) == 0 and f[i, j + 1] > f[i, j]:
                    f2[i, j] = 1
    return f2


# 边缘填充
def copyMakeBorder(f):
    rows, cols = f.shape
    newF = np.zeros((rows + 2, cols + 2))
    newF[0, 1:-1] = f[0, :]
    newF[1:-1, 1:-1] = f[0:, :]
    newF[rows, 1:-1] = f[rows - 1:]
    newF[:, 0] = newF[:, 1]
    newF[:, -1] = newF[:, -2]
    return newF


def show(f, s, a, b, c):
    plt.subplot(a, b, c)
    plt.imshow(f, "gray")
    plt.axis('on')
    plt.title(s)


def main():
    f = plt.imread("smallfingerprint.jpg")
    f_gray = cv.cvtColor(f, cv.COLOR_BGR2GRAY)
    hist1 = cv.calcHist([f_gray], [0], None, [256], [0, 256])

    topF = topHat(f_gray)
    hist2 = cv.calcHist([topF], [0], None, [256], [0, 256])
    # 原图与高帽后的图像
    plt.figure()
    show(f_gray, "original", 1, 2, 1)
    show(topF, "topF", 1, 2, 2)
    plt.show()
    # 原图与高帽后图像的直方图 确定50为阈值
    plt.figure()
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.subplot(1, 2, 1)
    plt.plot(hist1)
    plt.subplot(1, 2, 2)
    plt.plot(hist2)
    plt.xlim([0, 256])
    plt.show()

    # 二值化
    ret, binaryImg = cv.threshold(topF, 50, 255, cv.THRESH_BINARY)
    # 对指纹图像开闭运算运算减少周围噪声 进行开运算
    lockageonBinaryImg = lockageon(binaryImg)
    plt.figure()
    show(binaryImg, "binaryImg", 1, 2, 1)
    show(lockageonBinaryImg, "lockageonBinaryImg", 1, 2, 2)
    plt.show()

    skeletonF = lockageonBinaryImg.copy()
    skeletonF = refining(skeletonF)
    cutImg = cut(skeletonF)

    # 细化和裁剪输出
    plt.figure()
    show(skeletonF, "skeletonF", 1, 2, 1)
    show(cutImg, "cutImg", 1, 2, 2)
    plt.show()

    # 边缘和距离骨架提取
    contourImg = lockageonBinaryImg - imerode(lockageonBinaryImg, np.ones(9).reshape(3, 3))
    out = contourImg.copy()
    distanceImg = distance(contourImg)

    plt.figure()
    show(out, "contourImg", 1, 2, 1)
    show(distanceImg, "distanceImg", 1, 2, 2)
    plt.show()


if __name__ == '__main__':
    main()
