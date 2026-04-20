from math import atan2, cos, pi, sin
from copy import deepcopy
from utils import print_matrix, print_vector, get_column, get_vector_norm_c, mat_vec_mult, check_symmetric, max_off_diagonal_element, off_diagonal_norm

def jacobi_rotation_step(matrix, vectors):
    n = len(matrix)
    i, j, _ = max_off_diagonal_element(matrix)

    if abs(matrix[i][i] - matrix[j][j]) < 1e-12:
        angle = pi / 4
    else:
        angle = 0.5 * atan2(2 * matrix[i][j], matrix[i][i] - matrix[j][j])
    
    c, s = cos(angle), sin(angle)

    new_matrix = deepcopy(matrix)

    aii = matrix[i][i]
    ajj = matrix[j][j]
    aij = matrix[i][j]

    new_matrix[i][i] = c * c * aii + 2 * c * s * aij + s * s * ajj
    new_matrix[j][j] = s * s * aii - 2 * c * s * aij + c * c * ajj
    new_matrix[i][j] = new_matrix[j][i] = 0.0

    for k in range(n):
        if k != i and k != j:
            aik = matrix[i][k]
            ajk = matrix[j][k]

            new_matrix[i][k] = c * aik + s * ajk
            new_matrix[k][i] = new_matrix[i][k]

            new_matrix[j][k] = -s * aik + c * ajk
            new_matrix[k][j] = new_matrix[j][k]
    
    for k in range(n):
        vki = vectors[k][i]
        vkj = vectors[k][j]

        vectors[k][i] = c * vki + s * vkj
        vectors[k][j] = -s * vki + c * vkj
    
    return new_matrix, vectors

def jacobi_method(matrix, eps=1e-6, max_iterations=10000):
    n = len(matrix)
    current_matrix = deepcopy(matrix)
    vectors = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    errors = []

    for iteration in range(1, max_iterations + 1):
        current_error = off_diagonal_norm(current_matrix)
        errors.append(current_error)

        if current_error < eps:
            values = [current_matrix[i][i] for i in range(n)]
            return values, vectors, iteration-1, errors, current_matrix 
        current_matrix, vectors = jacobi_rotation_step(current_matrix, vectors)
    
    raise ValueError("Метод Якоби не сошелся за максимальное количество итераций")

def check_solution(original_matrix, values, vectors, eps=1e-6):
    n = len(original_matrix)

    print("Проверка решения:")
    for idx in range(n):
        eigenvector = get_column(vectors, idx)
        left = mat_vec_mult(original_matrix, eigenvector)
        right = [values[idx] * x for x in eigenvector]
        residual = [left[i] - right[i] for i in range(n)]
        residual_norm = get_vector_norm_c(residual)

        status = "OK" if residual_norm < eps else "FAIL"
        print(f"Собственное значение λ{idx + 1} = {values[idx]:.6f}")
        print(f"Собственный вектор x{idx + 1} = {[round(x, 6) for x in eigenvector]}")
        print()

    print("Проверка ортогональности собственных векторов:")
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_column(vectors, i)
            vj = get_column(vectors, j)
            dot_product = sum(vi[k] * vj[k] for k in range(n))
            status = "OK" if abs(dot_product) < eps else "FAIL"
            print(
                f"(x{i + 1}, x{j + 1}) = {dot_product:.12f} -> {status}"
            )
    print()

def main():
    a = [
        [9.0, -5.0, -6.0],
        [-5.0, 1.0, -8.0],
        [-6.0, -8.0, -3.0],
    ]
    eps = 1e-6

    check_symmetric(a)

    print_matrix(a, "Исходная матрица A")

    values, vectors, iterations, errors, diagonal_matrix = jacobi_method(a, eps)

    print(f"Число итераций: {iterations}")
    print()

    print_matrix(diagonal_matrix, "Матрица после метода вращений")
    print_vector(values, "Собственные значения:")
    print_matrix(vectors, "Матрица собственных векторов")
    print("Погрешность по итерациям:")
    for i, error in enumerate(errors):
        print(f"Итерация {i:3d}: {error:.12f}")
    print()

    check_solution(a, values, vectors, eps)


if __name__ == "__main__":
    main()
