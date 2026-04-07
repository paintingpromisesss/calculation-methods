from math import sqrt
EPS = 1e-12
def print_matrix(matrix, name="Matrix", precision=6, eps=EPS):
    cleaned = []
    for row in matrix:
        cleaned_row = []
        for x in row:
            if abs(x) < eps:
                x = 0.0
            cleaned_row.append(x)
        cleaned.append(cleaned_row)

    formatted = []
    for row in cleaned:
        formatted_row = []
        for x in row:
            formatted_row.append(f"{x:.{precision}f}")
        formatted.append(formatted_row)

    width = max(len(item) for row in formatted for item in row)

    print(f"{name}:")
    for row in formatted:
        print("[", end=" ")
        print("  ".join(f"{item:>{width}}" for item in row), end=" ")
        print("]")
    print()

def print_vector(vector, title, label="λ"):
    print(title)
    for i, value in enumerate(vector, start=1):
        print(f"{label}{i}: {value:12.6f}")
    print()

def get_vector_norm_c(vector):
    return max(abs(value) for value in vector)

def mat_vec_mult(matrix, vector):
    n = len(matrix)
    result = [0.0] * n
    for i in range(n):
        for j in range(n):
            result[i] += matrix[i][j] * vector[j]
    return result

def check_symmetric(matrix):
    n = len(matrix)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j] - matrix[j][i]) > 1e-12:
                raise ValueError(f"Матрица не симметрична: элемент ({i}, {j}) != элемент ({j}, {i})")

def get_column(matrix, col_index):
    return [matrix[i][col_index] for i in range(len(matrix))]

def max_off_diagonal_element(matrix):
    n = len(matrix)
    max_value = abs(matrix[0][1])
    max_i, max_j = 0, 1

    for i in range(n):
        for j in range(i + 1, n):
            if abs(matrix[i][j]) > max_value:
                max_value = abs(matrix[i][j])
                max_i, max_j = i, j

    return max_i, max_j, max_value

def off_diagonal_norm(matrix):
    n = len(matrix)
    norm = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            if i != j:
                norm += matrix[i][j] ** 2

    return sqrt(norm)