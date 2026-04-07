
from math import sqrt
EPS=1e-12

def get_hh_matrix(v):
    n = len(v)
    vv = sum(x * x for x in v)

    h = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                h[i][j] = 1.0 - 2.0 * v[i] * v[j] / vv
            else:
                h[i][j] = -2.0 * v[i] * v[j] / vv
    return h

def vector_norm(v):
    return sqrt(sum(x * x for x in v))


def matrix_multiply(a, b):
    rows = len(a)
    cols = len(b[0])
    inner = len(b)

    result = [[0.0] * cols for _ in range(rows)]
    for i in range(rows):
        for j in range(cols):
            s = 0.0
            for k in range(inner):
                s += a[i][k] * b[k][j]
            result[i][j] = s
    return result

def max_val_below_diagonal(matrix):
    n = len(matrix)
    max_val = 0.0

    for i in range(n):
        for j in range(i):
            val = abs(matrix[i][j])
            if val > max_val:
                max_val = val
    
    return max_val

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

def trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))


def determinant_3x3(matrix):
    a11, a12, a13 = matrix[0]
    a21, a22, a23 = matrix[1]
    a31, a32, a33 = matrix[2]

    return (
        a11 * a22 * a33
        + a12 * a23 * a31
        + a13 * a21 * a32
        - a13 * a22 * a31
        - a11 * a23 * a32
        - a12 * a21 * a33
    )

def check_solution(original_matrix, final_matrix, values, eps=1e-9):
    print("Проверка решения:")

    tr_a = trace(original_matrix)
    sum_lambda = sum(values)
    trace_ok = abs(tr_a - sum_lambda) < eps

    print(f"След матрицы:")
    print(f"tr(A)                 = {tr_a:.10f}")
    print(f"sum(lambda_i)         = {sum_lambda:.10f}")
    print(f"{'OK' if trace_ok else 'FAIL'}")
    print()

    if len(original_matrix) == 3:
        det_a = determinant_3x3(original_matrix)
        prod_lambda = 1.0
        for value in values:
            prod_lambda *= value

        det_ok = abs(det_a - prod_lambda) < eps

        print(f"Определитель матрицы:")
        print(f"det(A)                = {det_a:.10f}")
        print(f"prod(lambda_i)        = {prod_lambda:.10f}")
        print(f"{'OK' if det_ok else 'FAIL'}")
        print()

        max_below = max_val_below_diagonal(final_matrix)
        triangular_ok = max_below < eps
        print("Проверка верхнетреугольного вида:")
        print(f"Наибольшее значение под диагональю = {max_below:.10e}")
        print(f"{'OK' if triangular_ok else 'FAIL'}")
        print()
    print()