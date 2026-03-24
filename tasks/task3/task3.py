def max_norm(v1, v2):
    return max(abs(v1[i] - v2[i]) for i in range(len(v1)))


def simple_iteration(a, b, eps, max_iterations=1000):
    n = len(a)
    x_prev = [b[i] / a[i][i] for i in range(n)]
    alpha_norm = max(
        sum(abs(-a[i][j] / a[i][i]) for j in range(n) if j != i)
        for i in range(n)
    )

    for iteration in range(1, max_iterations + 1):
        x = [0.0] * n
        for i in range(n):
            x[i] = (b[i] - sum(a[i][j] * x_prev[j] for j in range(n) if j != i)) / a[i][i]

        diff = max_norm(x, x_prev)
        estimate = alpha_norm * diff / (1 - alpha_norm) if alpha_norm < 1 else diff
        if estimate < eps:
            return x, iteration

        x_prev = x

    raise ValueError("Метод простых итераций не сошелся.")


def seidel_method(a, b, eps, max_iterations=1000):
    n = len(a)
    x_prev = [b[i] / a[i][i] for i in range(n)]

    for iteration in range(1, max_iterations + 1):
        x = x_prev[:]
        for i in range(n):
            left = sum(a[i][j] * x[j] for j in range(i))
            right = sum(a[i][j] * x_prev[j] for j in range(i + 1, n))
            x[i] = (b[i] - left - right) / a[i][i]

        if max_norm(x, x_prev) < eps:
            return x, iteration

        x_prev = x

    raise ValueError("Метод Зейделя не сошелся.")


def print_vector(name, vector):
    print(name)
    for i, value in enumerate(vector, start=1):
        print(f"x{i} = {value:.10f}")
    print()


def main():
    a = [
        [-25.0, 4.0, -4.0, 9.0],
        [-9.0, 21.0, 5.0, -6.0],
        [9.0, 2.0, 19.0, -7.0],
        [-7.0, 4.0, -7.0, 25.0],
    ]
    b = [86.0, 29.0, 28.0, 68.0]
    eps = 1e-6

    x_iter, iter_count = simple_iteration(a, b, eps)
    x_seidel, seidel_count = seidel_method(a, b, eps)

    print("Задание 1.3, вариант 23\n")
    print(f"Точность: {eps}\n")

    print_vector("Метод простых итераций:", x_iter)
    print(f"Количество итераций: {iter_count}")
    print()

    print_vector("Метод Зейделя:", x_seidel)
    print(f"Количество итераций: {seidel_count}")


if __name__ == "__main__":
    main()
