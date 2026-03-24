from copy import deepcopy


def lu_decomposition(a):
    n = len(a)
    lu = deepcopy(a)
    p = list(range(n))
    swap_count = 0

    for k in range(n):
        pivot_row = max(range(k, n), key=lambda i: abs(lu[i][k]))
        if abs(lu[pivot_row][k]) < 1e-12:
            raise ValueError("Матрица вырождена")

        if pivot_row != k:
            lu[k], lu[pivot_row] = lu[pivot_row], lu[k]
            p[k], p[pivot_row] = p[pivot_row], p[k]
            swap_count += 1

        for i in range(k + 1, n):
            lu[i][k] /= lu[k][k]
            for j in range(k + 1, n):
                lu[i][j] -= lu[i][k] * lu[k][j]

    return lu, p, swap_count


def solve_lu(lu, p, b):
    n = len(lu)
    pb = [b[p[i]] for i in range(n)]

    y = [0.0] * n
    for i in range(n):
        y[i] = pb[i] - sum(lu[i][j] * y[j] for j in range(i))

    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - sum(lu[i][j] * x[j] for j in range(i + 1, n))) / lu[i][i]

    return x


def determinant(lu, swap_count):
    det = 1.0
    for i in range(len(lu)):
        det *= lu[i][i]
    return -det if swap_count % 2 else det


def inverse_matrix(lu, p):
    n = len(lu)
    inv = [[0.0] * n for _ in range(n)]

    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0
        x = solve_lu(lu, p, e)
        for row in range(n):
            inv[row][col] = x[row]

    return inv


def print_vector(name, vector):
    print(name)
    for i, value in enumerate(vector, start=1):
        print(f"x{i} = {value:.10f}")
    print()


def print_matrix(name, matrix):
    print(name)
    for row in matrix:
        print(" ".join(f"{value: .10f}" for value in row))
    print()


def main():
    a = [
        [2.0, -7.0, 8.0, -4.0],
        [0.0, -1.0, 4.0, -1.0],
        [3.0, -4.0, 2.0, -1.0],
        [-9.0, 1.0, -4.0, 6.0],
    ]
    b = [57.0, 24.0, 28.0, 12.0]

    lu, p, swap_count = lu_decomposition(a)
    x = solve_lu(lu, p, b)
    det = determinant(lu, swap_count)
    inv = inverse_matrix(lu, p)

    print("Задание 1.1, вариант 23\n")
    print_vector("Решение системы:", x)
    print(f"Определитель: {det:.10f}\n")
    print_matrix("Обратная матрица:", inv)


if __name__ == "__main__":
    main()
