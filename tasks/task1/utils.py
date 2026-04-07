EPS=1e-12
def print_matrix(matrix, name="Matrix", precision=6):
    cleaned = []
    for row in matrix:
        cleaned_row = []
        for x in row:
            if abs(x) < EPS:
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

def mat_vec_mult(matrix, vector):
    n = len(matrix)
    return [sum(matrix[i][j] * vector[j] for j in range(n)) for i in range(n)]

def mat_mat_mult(A, B):
    n = len(A)
    m = len(B[0])
    p = len(B)

    result = [[0.0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            result[i][j] = sum(A[i][k] * B[k][j] for k in range(p))

    return result