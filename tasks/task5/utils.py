from cmath import sqrt
from math import prod
EPS=1e-12

def get_hh_matrix(v, eps=EPS):
    n = len(v)
    vv = sum(abs(x) ** 2 for x in v)

    if vv < eps:
        return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    h = [[0j] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            delta = 1.0 if i == j else 0.0
            h[i][j] = delta - 2.0 * v[i] * v[j].conjugate() / vv

    return h

def vector_norm(v):
    return sqrt(sum(abs(x) ** 2 for x in v))


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

def format_number(x, precision=6, eps=EPS):
    if isinstance(x, complex):
        real = 0.0 if abs(x.real) < eps else x.real
        imag = 0.0 if abs(x.imag) < eps else x.imag

        if imag == 0.0:
            return f"{real:.{precision}f}"

        if real == 0.0:
            return f"{imag:.{precision}f}i"

        sign = "+" if imag >= 0 else "-"
        return f"{real:.{precision}f} {sign} {abs(imag):.{precision}f}i"

    if abs(x) < eps:
        x = 0.0

    return f"{x:.{precision}f}"

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
            formatted_row.append(format_number(x, precision, eps))
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
        print(f"{label}{i}: {format_number(value)}")
    print()

def trace(matrix):
    return sum(matrix[i][i] for i in range(len(matrix)))

def is_quasi_triangular(matrix, eps=EPS):
    n = len(matrix)

    for i in range(n):
        for j in range(i-1):
            if abs(matrix[i][j]) > eps:
                return False
    return True

def eigenvalues_from_quasi_triangular(matrix, eps=EPS):
    n = len(matrix)
    values = []

    i = 0
    while i < n:
        if i < n - 1 and abs(matrix[i+1][i]) > eps:
            a11 = matrix[i][i]
            a12 = matrix[i][i+1]
            a21 = matrix[i+1][i]
            a22 = matrix[i+1][i+1]

            tr = a11 + a22
            det = a11 * a22 - a12 * a21
            disc = tr * tr - 4 * det

            root  = sqrt(disc)
            values.append((tr + root) / 2)
            values.append((tr - root) / 2)

            i += 2
        else:
            values.append(matrix[i][i])
            i += 1
    
    return values


def determinant(matrix):
    n = len(matrix)
    a = [row[:] for row in matrix]
    det = 1 + 0j

    for i in range(n):
        pivot = i

        for row in range(i + 1, n):
            if abs(a[row][i]) > abs(a[pivot][i]):
                pivot = row

        if abs(a[pivot][i]) < EPS:
            return 0.0

        if pivot != i:
            a[i], a[pivot] = a[pivot], a[i]
            det *= -1

        pivot_value = a[i][i]
        det *= pivot_value

        for row in range(i + 1, n):
            factor = a[row][i] / pivot_value

            for col in range(i + 1, n):
                a[row][col] -= factor * a[i][col]

    return det

def check_solution(original_matrix, final_matrix, values, eps=1e-7):
    print("Проверка решения:")

    original_trace = trace(original_matrix)
    final_trace = trace(final_matrix)
    eigen_trace = sum(values)

    original_det = determinant(original_matrix)
    eigen_det = prod(values, start=1 + 0j)


    print(f"След исходной матрицы:              {format_number(original_trace)}")
    print(f"След матрицы после QR-алгоритма:    {format_number(final_trace)}")
    print(f"Сумма найденных собственных значений: {format_number(eigen_trace)}")
    print()
    trace_diff = abs(original_trace - eigen_trace)

    print(f"|tr(A) - Σλ|: {trace_diff:.10e}")

    if trace_diff < eps:
        print("Проверка по следу: пройдена")
    else:
        print("Проверка по следу: не пройдена")

    print()

    print(f"Определитель исходной матрицы:      {format_number(original_det)}")
    print(f"Произведение собственных значений:  {format_number(eigen_det)}")
    print()

    if is_quasi_triangular(final_matrix, eps):
        print("Финальная матрица имеет верхнюю квазитреугольную форму: да")
    else:
        print("Финальная матрица имеет верхнюю квазитреугольную форму: нет")

    print()