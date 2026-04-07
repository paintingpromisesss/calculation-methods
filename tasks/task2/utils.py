EPS = 1e-12

def print_vector(name, vector, prefix):
    print(name)
    for i, value in enumerate(vector, start=1):
        print(f"{prefix}{i} = {value:.10f}")
    print()


def check_solution(a, b, c, d, x):
    n = len(x)
    print("Проверка:")
    for i in range(n):
        left = b[i] * x[i]
        if i > 0:
            left += a[i] * x[i - 1]
        if i < n - 1:
            left += c[i] * x[i + 1]

        ok = abs(left - d[i]) < EPS
        print(
            f"Уравнение {i+1}: "
            f"левая часть = {left:.10f}, "
            f"правая часть = {d[i]:.10f}, "
            f"{'OK' if ok else 'FAIL'}"
        )

