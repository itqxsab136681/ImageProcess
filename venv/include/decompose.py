import numpy as np

np.set_printoptions(suppress=True)


def matrix_decomposition(A, method):
    """矩阵分解，可选参数为G,S,H,L"""
    if method == '1':
        return Givens(A)
    elif method == '2':
        return GramSchmidt(A)
    elif method == '3':
        return householder(A)
    elif method == '4':
        return LU(A)


def Givens(A):
    """Givens变换"""
    (r, c) = np.shape(A)
    Q = np.identity(r)
    R = np.copy(A)
    (rows, cols) = np.tril_indices(r, -1, c)
    for (row, col) in zip(rows, cols):
        if R[row, col] != 0:
            r_ = np.hypot(R[col, col], R[row, col])  # d
            c = R[col, col] / r_
            s = -R[row, col] / r_
            G = np.identity(r)
            G[[col, row], [col, row]] = c
            G[row, col] = s
            G[col, row] = -s
            R = np.dot(G, R)  # R=G(n-1,n)*...*G(2n)*...*G(23,1n)*...*G(12)*A
            Q = np.dot(Q, G.T)  # Q=G(n-1,n).T*...*G(2n).T*...*G(23,1n).T*...*G(12).T
    return Q, R


def GramSchmidt(A):
    """施密特正交化"""
    m, n = A.shape
    temp_A = np.zeros([m, n])
    r_list = [0.0] * n
    r_list[0] = np.sqrt(np.inner(A[:, 0], A[:, 0]))
    temp_A[:, 0] = A[:, 0] / r_list[0]
    for i in range(1, n):
        temp_q = A[:, i].copy()
        for j in range(i):
            temp_q = temp_q - np.inner(temp_A[:, j], temp_q) * temp_A[:, j]
        r_list[i] = np.sqrt(np.inner(temp_q, temp_q))
        temp_A[:, i] = temp_q / r_list[i]
    R = np.zeros([n, n])
    for i in range(n):
        for j in range(i, n):
            R[i, i] = r_list[i]
            R[i, j] = np.inner(A[:, j], temp_A[:, i])
    return temp_A, R


def householder(A):
    """Householder变换"""
    m, n = np.shape(A)
    Q = np.identity(m)
    R = np.copy(A)
    for i in range(m - 1):
        x = R[i:, i]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x)
        u = x - e
        v = u / np.linalg.norm(u)
        Q_temp = np.identity(n)
        Q_temp[i:, i:] -= 2.0 * np.outer(v, v)
        R = np.dot(Q_temp, R)  # R=H(n-1)*...*H(2)*H(1)*A
        Q = np.dot(Q, Q_temp)  # Q=H(n-1)*...*H(2)*H(1)  H为自逆矩阵
    return Q, R


def LU(C):
    '''LU分解'''
    B = C.copy()
    A = C.copy()
    m, n = A.shape
    # print(A)

    L = np.zeros(shape=(n, n))
    U = np.zeros(shape=(n, n))

    for k in range(n - 1):
        vector = A[:, k]
        if vector[k] != 0:
            vector[k + 1:] = vector[k + 1:] / vector[k]
            vector[0:k + 1] = np.zeros(k + 1)
            L[:, k] = vector
            L[k, k] = 1.0
            for l in range(k + 1, n):
                B[l, :] = B[l, :] - vector[l] * B[k, :]

            A = np.array(B)
        else:
            A[[k, k + 1], :] = A[[k + 1, k], :]
            B[[k, k + 1], :] = B[[k + 1, k], :]
            vector = A[:, k]
            vector[k + 1:] = vector[k + 1:] / vector[k]
            vector[0:k + 1] = np.zeros(k + 1)
            L[:, k] = vector
            L[k, k] = 1.0
            for l in range(k + 1, n):
                B[l, :] = B[l, :] - vector[l] * B[k, :]

            A = np.array(B)
    L[k + 1][k + 1] = 1.0
    U = A
    return L, U


def main():
    M = input("请输入矩阵按行输入，以空格分割enter键结束:")
    M = M.split(" ")
    n = int(len(M) ** 0.5)
    # 生成n阶矩阵
    A = np.array(M).astype(np.float)
    A = A.reshape(n, n)
    while True:
        method = input('请输入分解形式（Givens输入字母1,Householder输入字母2,施密特正交化输入字母3,LU分解输入字母4）:')
        list1 = ['1', '2', '3', '4']
        if method in list1:
            Q, R = matrix_decomposition(A, method)
            if method == '1':
                print(method + ':')
                print('L=')
                print(Q)
                print('U=')
                print(R)
            else:
                print(method + ':')
                print('Q=')
                print(Q)
                print('R=')
                print(R)
        else:
            print('输入错误!请按指定格式输入！')
        flag1 = input('是否继续选择其他分解方式：yes/no:')
        if flag1 == 'yes':
            continue
        else:
            break


main()
