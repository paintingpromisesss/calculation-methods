EPS = 1e-12


def get_matrix_norm_c(matrix):
    return max(sum(abs(value) for value in row) for row in matrix)


def get_vector_norm_c(vector):
    return max(abs(value) for value in vector)


def print_vector(name, vector, prefix):
    print(name)
    for i, value in enumerate(vector, start=1):
        print(f"{prefix}{i} = {value:.10f}")
    print()


def check_solution(a, b, x):
    print("Проверка Ax = b:")
    for i in range(len(a)):
        left = sum(a[i][j] * x[j] for j in range(len(x)))
        print(f"строка {i + 1}: {left:.10f} ≈ {b[i]:.10f}")
    print()


def fix_zero_diagonal(matrix, b, eps):
    n = len(matrix)
    for i in range(n):
        if abs(matrix[i][i]) < eps:
            swapped = False
            for k in range(i + 1, n):
                if abs(matrix[k][i]) > eps:
                    matrix[i], matrix[k] = matrix[k], matrix[i]
                    b[i], b[k] = b[k], b[i]
                    swapped = True
                    break
            if not swapped:
                raise ValueError(
                    f"Не удалось устранить нулевой элемент на диагонали в столбце {i}")

    return matrix, b


def prepare_system(a, b, eps, n):
    alpha = [[0.0] * n for _ in range(n)]
    beta = [0.0] * n

    for i in range(n):
        if abs(a[i][i]) < eps:
            raise ValueError(f"Нулевой элемент на диагонали в строке {i}")

        beta[i] = b[i] / a[i][i]
        for j in range(n):
            if i != j:
                alpha[i][j] = -a[i][j] / a[i][i]
    return alpha, beta


def lower_triangular_matrix(matrix):
    n = len(matrix)
    lower = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            lower[i][j] = matrix[i][j]
    return lower


def upper_triangular_matrix(matrix):
    n = len(matrix)
    upper = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            upper[i][j] = matrix[i][j]
    return upper


def split_alpha(alpha):
    n = len(alpha)
    lower = lower_triangular_matrix(alpha)
    upper = upper_triangular_matrix(alpha)
    return lower, upper
