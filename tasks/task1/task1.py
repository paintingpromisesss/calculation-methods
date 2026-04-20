from copy import deepcopy

from utils import print_matrix, EPS, mat_vec_mult, mat_mat_mult


def determinant_from_lu(U, swap_count):
    det = 1.0
    n = len(U)
    for i in range(n):
        det *= U[i][i]
    if swap_count % 2 == 1:
        det = -det
    return det


def inverse_from_lu(P, L, U):
    n = len(U)
    inv = [[0.0] * n for _ in range(n)]
    for col in range(n):
        e = [0.0] * n
        e[col] = 1.0

        Pb = mat_vec_mult(P, e)
        y = forward_solve(L, Pb)
        x = backward_solve(U, y)

        for row in range(n):
            inv[row][col] = x[row]

    return inv


def get_lu(a):
    n = len(a)
    U = deepcopy(a)
    P = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    L = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
    swap_count = 0

    for i in range(n):
        max_elem_row = max(range(i, n), key=lambda r: abs(U[r][i]))

        if abs(U[max_elem_row][i]) < EPS:
            raise ValueError("Матрица вырождена или почти вырождена")

        if max_elem_row != i:
            U[i], U[max_elem_row] = U[max_elem_row], U[i]
            P[i], P[max_elem_row] = P[max_elem_row], P[i]
            swap_count += 1

            for k in range(i):
                L[i][k], L[max_elem_row][k] = L[max_elem_row][k], L[i][k]

        for j in range(i + 1, n):
            factor = U[j][i] / U[i][i]
            L[j][i] = factor

            for k in range(i, n):
                U[j][k] -= factor * U[i][k]

    return P, L, U, swap_count


def forward_solve(L, b):
    n = len(L)
    y = [0.0] * n

    for i in range(n):
        if abs(L[i][i]) < EPS:
            raise ValueError(f"Нулевой или почти нулевой элемент L[{i}][{i}]")
        y[i] = b[i]
        for j in range(i):
            y[i] -= L[i][j] * y[j]
        y[i] /= L[i][i]

    return y


def backward_solve(U, y):
    n = len(U)
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        if abs(U[i][i]) < EPS:
            raise ValueError(f"Нулевой или почти нулевой элемент U[{i}][{i}]")
        x[i] = y[i]
        for j in range(i + 1, n):
            x[i] -= U[i][j] * x[j]
        x[i] /= U[i][i]
    return x


def main():
    a = [
        [2.0, -7.0, 8.0, -4.0],
        [0.0, -1.0, 4.0, -1.0],
        [3.0, -4.0, 2.0, -1.0],
        [-9.0, 1.0, -4.0, 6.0],
    ]
    b = [57.0, 24.0, 28.0, 12.0]

    P, L, U, swap_count = get_lu(a)
    print_matrix(U, "Матрица U")
    print_matrix(L, "Матрица L")
    print_matrix(P, "Матрица P")

    Pb = mat_vec_mult(P, b)
    y = forward_solve(L, Pb)
    x = backward_solve(U, y)
    print("Pb =", [f"{v:.6f}" for v in Pb])
    print("y  =", [f"{v:.6f}" for v in y])
    print("x  =", [f"{v:.6f}" for v in x])
    print()

    det_a = determinant_from_lu(U, swap_count)
    print(f"det(A) = {det_a:.6f}")
    print()
    A_inv = inverse_from_lu(P, L, U)
    print_matrix(A_inv, "A^(-1)")

    print()
    check = mat_mat_mult(a, A_inv)
    print_matrix(check, "A * A^(-1)")


if __name__ == "__main__":
    main()
