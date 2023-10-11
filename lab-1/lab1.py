import numpy as np
from numpy import *
import math

def double_simplex(c, b, a_matrix, j_vector):
    m, n = a_matrix.shape
    iter_count = 1
    j_vector -= 1
    y = get_initial_y(c, a_matrix, j_vector)
    x_0 = [0 for _ in range(n)]

    print('\n\n===========Double simplex===============')

    while True:

        not_J = delete(arange(n), j_vector)
        B = linalg.inv(a_matrix[:, j_vector])
        kappa = B.dot(b)

        if all(kappa >= 0):
            for j, _kappa in zip(j_vector, kappa):
                x_0[j] = _kappa

            print('\n****************************************')

            print("Количество итераций : ", iter_count)
            print("Максимальная прибыль : ", c.dot(x_0))
            print(list(map(lambda _x: round(float(_x), 3), list(x_0))), "-  план")
            print('****************************************\n')

            print('===========Double simplex ends==========\n\n')

            return list(map(lambda _x: round(float(_x), 3), list(x_0)))

        k = argmin(kappa)
        delta_y = B[k]
        mu = delta_y.dot(a_matrix)

        sigma = []
        for i in not_J:
            if mu[i] >= 0:
                sigma.append(inf)

            else:
                sigma.append((c[i] - a_matrix[:, i].dot(y)) / mu[i])

        sigma_0_ind = not_J[argmin(sigma)]
        sigma_0 = min(sigma)

        if sigma_0 == inf:
            print("Задача не имеет решения, т.к. пусто множество ее допустимых планов.")
            print('===========Double simplex ends==========\n\n')
            return "Задача не имеет решения"

        y += sigma_0 * delta_y
        j_vector[k] = sigma_0_ind
        iter_count += 1

    


def get_initial_y(c, a_matrix, j_vector):
    return (c[j_vector]).dot(linalg.inv(a_matrix[:, j_vector]))

def parse_first_conditions(c_0, A_0, d_minus, d_plus):

    for i in range(0, len(c_0)):
        if(c_0[i] > 0 ):

            c_0[i] *= -1

            for j in range(0, len(A_0[0])):
                A_0[j][i] *= -1

            d_minus[i] *= -1
            d_plus[i] *= -1

            k = d_minus[i]
            d_minus[i] = d_plus[i]
            d_plus[i] = k

    return c_0, A_0, d_minus, d_plus


def main():

    c_0 = np.array([1, 1])
    A_0 = np.array([[5, 9], [9, 5]])
    b_0 = np.array([63,63])
    d_o_minus = np.array([1,1])
    d_o_plus= np.array([6,6])
    x_0 = np.array([0,0])

    c_init = np.copy(c_0)

    c_0, A_0, d_o_minus, d_o_plus = parse_first_conditions(c_0, A_0, d_o_minus, d_o_plus)

    print("C = ", c_0) 
    print("A =\n", A_0)
    print("d_minus = ", d_o_minus)
    print("d_plus = ", d_o_plus)

    n = len(A_0[0])
    m = len(A_0)

    x_1 = np.concatenate([x_0, [0]* (len(A_0[0]) + len(A_0))])
    c_1 = np.concatenate([c_0, [0]* (len(A_0[0]) + len(A_0))])
    b_1  = np.concatenate([b_0, d_o_plus])
    d_1_minus = np.concatenate([d_o_minus, [0]* (len(A_0[0]) + len(A_0))])
    A_0 = np.hstack([np.vstack([A_0, np.eye(len(A_0[0]))]), np.eye(len(A_0[0]) + len(A_0))])

    print("\n\n\nNew X = ", x_1)
    print("New C = ", c_1)
    print("New b = ", b_1)
    print("New d_minus = ", d_1_minus)
    print("New A =\n", A_0)

    x_star = []
    r = 0
    S = []
    alpha = 0


    S =[]
    S.append([d_1_minus, b_1, alpha, d_1_minus])
    iter_count = 0

    while S:
        iter_count = iter_count + 1
        print("\n\nIteration #", iter_count)
        D = S.pop()

        alpha_shtr = D[2] + np.dot(c_1, D[0])
        b_shtr = np.array(D[1] - np.dot(A_0, D[0]))

        j = np.array([0] * (n + m))
        for i in range (0, len(j)):
            j[i] =  n  + i + 1

        print("\nDelta =", D[3])
        print("Alfa =", alpha_shtr)
        print("New b =", b_shtr)

        x_voln = double_simplex(c_1, b_shtr, A_0, j)    
        print("Simplex_res = ", x_voln)
        if x_voln == "Задача не имеет решения":
            continue

        flg = 1
        for i in range(0, len(x_voln)):
            if not (float(x_voln[i]) == int(x_voln[i])):
                flg = 0
                if (not x_star) or (math.floor(np.dot(c_1, x_voln)) > r):
                    b_two_shtr = np.copy(b_shtr)
                    b_two_shtr[m + i] = math.floor(x_voln[i])
                    d_zero = np.array([0] * (2*n + m))
                    S.append([d_zero, b_two_shtr, alpha_shtr, D[3]])
                    d_zero[i] = math.ceil(x_voln[i])
                    S.append([d_zero, b_shtr, alpha_shtr, D[3] + d_zero])
                break

        if flg == 1:
            x_final = x_voln + D[3]
            if (not x_star) or (np.dot(c_1, x_final) + alpha_shtr > r):
                x_star = x_final
                r = np.dot(c_1, x_final) + alpha_shtr

    print("\n\nEND\n")
    print("X_Star =", x_star)
    print("R = ", r)

    for i in range(0, len(x_0)):
        if c_init[i] < 0:
            x_0[i] = x_star[i]
        else:
            x_0[i] = -x_star[i]

    print("\nPlan = ", x_0)

if __name__ == "__main__":
    main()



