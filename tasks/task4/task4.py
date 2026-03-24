from math import atan, cos, pi, sin, sqrt


def identity_matrix(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def max_offdiag_index(a):
    n = len(a)
    i_max, j_max = 0, 1
    max_value = abs(a[0][1])

    for i in range(n):
        for j in range(i + 1, n):
            if abs(a[i][j]) > max_value:
                max_value = abs(a[i][j])
                i_max, j_max = i, j

    return i_max, j_max


def offdiag_norm(a):
    s = 0.0
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            s += a[i][j] ** 2
    return sqrt(s)


def matrix_mul(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)
    return [
        [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(cols)]
        for i in range(rows)
    ]


def transpose(a):
    return [[a[j][i] for j in range(len(a))] for i in range(len(a[0]))]


def jacobi_rotation_method(a, eps, max_iterations=100):
    n = len(a)
    a_k = [row[:] for row in a]
    vectors = identity_matrix(n)
    history = []

    for iteration in range(1, max_iterations + 1):
        i, j = max_offdiag_index(a_k)
        if a_k[i][i] == a_k[j][j]:
            phi = pi / 4
        else:
            phi = 0.5 * atan(2 * a_k[i][j] / (a_k[i][i] - a_k[j][j]))

        c = cos(phi)
        s = sin(phi)

        u = identity_matrix(n)
        u[i][i] = c
        u[j][j] = c
        u[i][j] = -s
        u[j][i] = s

        a_k = matrix_mul(transpose(u), matrix_mul(a_k, u))
        vectors = matrix_mul(vectors, u)

        error = offdiag_norm(a_k)
        history.append((iteration, error))
        if error < eps:
            eigenvalues = [a_k[k][k] for k in range(n)]
            eigenvectors = [[vectors[row][col] for row in range(n)] for col in range(n)]
            return eigenvalues, eigenvectors, history

    raise ValueError("Метод вращений не сошелся.")


def print_vector(name, vector):
    print(name)
    for i, value in enumerate(vector, start=1):
        print(f"x{i} = {value:.10f}")
    print()


def main():
    a = [
        [9.0, -5.0, -6.0],
        [-5.0, 1.0, -8.0],
        [-6.0, -8.0, -3.0],
    ]
    eps = 1e-6

    eigenvalues, eigenvectors, history = jacobi_rotation_method(a, eps)

    print("Задание 1.4, вариант 23\n")
    print(f"Точность: {eps}\n")

    print("Собственные значения:")
    for i, value in enumerate(eigenvalues, start=1):
        print(f"lambda{i} = {value:.10f}")
    print()

    print("Собственные векторы:")
    for i, vector in enumerate(eigenvectors, start=1):
        print_vector(f"v{i}:", vector)

    print(f"Количество итераций: {len(history)}\n")
    print("Погрешность по итерациям:")
    for iteration, error in history:
        print(f"{iteration}: {error:.10f}")


if __name__ == "__main__":
    main()
