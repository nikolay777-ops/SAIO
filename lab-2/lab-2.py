import numpy as np


def dot(A, B, index):
    C = np.zeros(A.shape)
    for i in range(len(A)):
        for j in range(len(A)):
            C[i][j] += A[i][index] * B[index][j]
            if i != index:
                C[i][j] += B[i][j]

    return C


def find_inv(A_inv: np.array, x: np.array, index: int):
    # 1:
    l = A_inv @ x
    if l[index] == 0:
        return False

    # 2:
    l_wave = np.copy(l)
    l_wave[index] = -1.

    # 3:
    l_hat = -1. / l[index] * l_wave

    # 4:
    Q = np.identity(len(x))
    Q[:, index] = l_hat

    # 5:
    return dot(Q, A_inv, index)


def solve_simplex(c: np.array, A: np.array, x: np.array, B=None):
    i = 0
    while True:
        if i == 0 and B is None:
            B = np.nonzero(x)[0]
        # 1
        A_B = A[:, B]
        if i == 0:
            A_B_inv = np.linalg.inv(A_B)
        else:
            A_B_inv = find_inv(A_B_inv, A[:, j_0], k)

        # 2
        c_B = c[B]

        # 3
        u = c_B @ A_B_inv

        # 4
        delta = u @ A - c

        # 5
        if (delta >= 0).all():
            return x, B

        # 6
        j_0 = np.argmax(delta < 0)

        # 7
        z = A_B_inv @ A[:, j_0]

        # 8
        theta = np.empty(len(z))
        for i in range(len(z)):
            if z[i] > 0:
                theta[i] = x[B[i]] / z[i]
            else:
                theta[i] = np.Inf

        # 9
        theta_0 = np.min(theta)

        # 10
        if theta_0 == np.inf:
            raise ValueError('Целевой функционал задачи не ограничен сверху на множестве допустимых планов')

        # 11
        k = np.argmin(theta)
        j_asterisk = B[k]

        # 12
        B[k] = j_0

        # 13
        x[j_0] = theta_0
        for i in range(len(B)):
            if i == k:
                continue
            x[B[i]] -= theta_0 * z[i]
        x[j_asterisk] = 0

        i += 1


def solve_simplex_initial(c: np.array, A: np.array, b: np.array):
    n = len(c)
    m = len(b)
    assert A.shape == (m, n)

    # 1
    for i in range(len(b)):
        if b[i] < 0:
            b[i] *= -1
            A[i] *= -1

    # 2
    c_wave = np.array([0] * n + [-1] * m)
    x_wave = np.zeros(n + m)
    A_wave = np.hstack((A, np.eye(A.shape[0])))

    # 3
    x_wave[n:] = b
    B = np.array([i for i in range(n, n + m)])

    # 4
    x_wave, B = solve_simplex(c_wave, A_wave, x_wave, B)

    # 5
    if not (x_wave[n:] == 0).all():
        raise ValueError('Задача несовместна')

    # 6
    x = x_wave[:n]

    while True:
        # 7
        if (B < n - 1).all():
            return x

        # 8
        k = B.argmax()
        j_k = B[k]
        i = j_k - n

        # 9
        l = np.ones([n, m])
        l[:, :] = np.nan

        for j in range(n):
            if j not in B:
                l[j] = np.linalg.inv(A_wave[:, B]) @ A_wave[:, j]

        # 10
        found_nonzero = False
        for j, l_j in enumerate(l):
            if not np.isnan(l_j[k]) and l_j[k] != 0:
                B[k] = j
                found_nonzero = True
                break

        if not found_nonzero:
            B = np.delete(B, k)

            A = np.delete(A, i, axis=0)
            A_wave = np.delete(A_wave, i, axis=0)
            b = np.delete(b, i)

            A_wave = np.delete(A_wave, n + i, axis=1)
            c_wave = np.delete(c_wave, n + i)


# c = np.array([0, 1, 0, 0])
# A = np.array([
    # [3, 2, 1, 0],
    # [-3, 2, 0, 1]
# ])
# b = np.array([6, 0])

c = np.array([2, 1, -3])  
A = np.array([[1, -1, 1], [3, 1, -1]]) 
b = np.array([1, 2])  


def solve(c, A, b):

    # c (коэффициенты целевой функции), A (матрица коэффициентов ограничений) и b (вектор правых частей ограничений).

    x = solve_simplex_initial(c, A, b)
    x_is_int = np.isclose(x, x.astype(int), 10 ** -3)

    if np.all(x_is_int):
        print(x)
        return x

    x, B = solve_simplex(c, A, x)

    # Здесь x обновляется, и также вычисляется множество базисных индексов B.

    N = [i for i in range(len(x)) if i not in B]

    # Создание списка N, который содержит индексы переменных, не входящих в базис. Эти индексы представляют небазисные переменные.

    A_B = A[:, B]

    # Извлечение из матрицы A только тех столбцов, которые соответствуют базисным переменным.

    A_N = A[:, N]

    # Извлечение из матрицы A только тех столбцов, которые соответствуют небазисным переменным.

    mul = np.linalg.inv(A_B) @ A_N

    # Вычисление произведения обратной матрицы базисных переменных на матрицу небазисных переменных.

    float_idx = np.argmin(x_is_int)

    # Нахождение индекса первой небазисной переменной, которая не является целым числом.

    k = np.argmax(B == float_idx)

    # Нахождение индекса переменной в базисе, которая соответствует найденной небазисной переменной.


    l = mul[k, :]

    # Извлечение строки l из матрицы mul, которая соответствует найденной базисной переменной.

    l_floats = l - np.floor(l)

    l_result = np.zeros(len(x) + 1)

    # Создание нулевого вектора l_result длиной, равной длине вектора x плюс один элемент для добавления новой переменной.

    l_result[-1] = -1

    l_result[N] = l_floats
    print(l_result, x[k])
    return l_result, x[k]


solve(c, A, b)

