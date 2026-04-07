from math import sqrt
from utils import get_hh_matrix, vector_norm, matrix_multiply, max_val_below_diagonal, print_matrix, print_vector, check_solution


def qr_decomposition(matrix):
    n = len(matrix)
    q = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    r = [row[:] for row in matrix]

    for k in range(n-1):
        x = [r[i][k] for i in range(k, n)]

        x_norm = vector_norm(x)
        if x_norm == 0:
            continue

        sign = 1.0 if x[0] >= 0 else -1.0

        e1 = [0.0] * len(x)
        e1[0] = 1.0

        u = [x[i] + sign * x_norm * e1[i] for i in range(len(x))]

        v = u[:]

        h_small = get_hh_matrix(v)

        h = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
        for i in range(k, n):
            for j in range(k, n):
                h[i][j] = h_small[i-k][j-k]

        r = matrix_multiply(h, r)
        q = matrix_multiply(q, h)

    return q, r


def qr_algorithm(matrix, eps, max_iterations=10000):
    a_k = [row[:] for row in matrix]
    n = len(a_k)

    for iteration in range(max_iterations):
        q, r = qr_decomposition(a_k)
        a_k = matrix_multiply(r, q)

        if max_val_below_diagonal(a_k) < eps:
            values = [a_k[i][i] for i in range(n)]
            return values, a_k, iteration+1

    value = [a_k[i][i] for i in range(n)]
    return value, a_k, max_iterations


def main():
    a = [
        [1.0, 5.0, -6.0],
        [9.0, -7.0, -9.0],
        [6.0, -1.0, -9.0]
    ]
    eps = 1e-12

    values, final_matrix, iterations = qr_algorithm(a, eps)
    print_matrix(a, "Исходная матрица", eps=eps)
    print_matrix(final_matrix, "Матрица после QR-алгоритма", eps=eps)

    print(f"Число итераций: {iterations}")
    print_vector(values, "Собственные значения:", "λ")

    check_solution(a, final_matrix, values)


if __name__ == "__main__":
    main()