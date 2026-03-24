from math import sqrt


def qr_decomposition(a):
    n = len(a)
    cols = [[a[i][j] for i in range(n)] for j in range(n)]
    q_cols = []
    r = [[0.0] * n for _ in range(n)]

    for j in range(n):
        v = cols[j][:]
        for i in range(j):
            r[i][j] = sum(q_cols[i][k] * cols[j][k] for k in range(n))
            for k in range(n):
                v[k] -= r[i][j] * q_cols[i][k]

        r[j][j] = sqrt(sum(v[k] * v[k] for k in range(n)))
        q_cols.append([v[k] / r[j][j] for k in range(n)])

    q = [[q_cols[j][i] for j in range(n)] for i in range(n)]
    return q, r


def matrix_mul(a, b):
    return [
        [sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
        for i in range(len(a))
    ]


def lower_norm(a):
    s = 0.0
    for i in range(1, len(a)):
        for j in range(min(i, len(a[0]))):
            s += abs(a[i][j])
    return s


def qr_algorithm(a, eps, max_iterations=1000):
    a_k = [row[:] for row in a]

    for iteration in range(1, max_iterations + 1):
        q, r = qr_decomposition(a_k)
        a_k = matrix_mul(r, q)

        if lower_norm(a_k) < eps:
            eigenvalues = [a_k[i][i] for i in range(len(a_k))]
            return q, r, eigenvalues, iteration

    raise ValueError("QR-алгоритм не сошелся.")


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print(" ".join(f"{value: .10f}" for value in row))
    print()


def main():
    a = [
        [1.0, 5.0, -6.0],
        [9.0, -7.0, -9.0],
        [6.0, -1.0, -9.0],
    ]
    eps = 1e-6

    q, r, eigenvalues, iterations = qr_algorithm(a, eps)

    print("Задание 1.5, вариант 23\n")
    print(f"Точность: {eps}\n")

    print_matrix("Q-матрица:", q)
    print_matrix("R-матрица:", r)

    print("Собственные значения:")
    for i, value in enumerate(eigenvalues, start=1):
        print(f"lambda{i} = {value:.10f}")
    print()

    print(f"Количество итераций QR-алгоритма: {iterations}")


if __name__ == "__main__":
    main()
