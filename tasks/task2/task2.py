def thomas_algorithm(a, b, c, d):
    n = len(b)
    p = [0.0] * n
    q = [0.0] * n

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] + a[i] * p[i - 1]
        if abs(denominator) < 1e-12:
            raise ValueError("Нулевой знаменатель")

        p[i] = 0.0 if i == n - 1 else -c[i] / denominator
        q[i] = (d[i] - a[i] * q[i - 1]) / denominator

    x = [0.0] * n
    x[-1] = q[-1]

    for i in range(n - 2, -1, -1):
        x[i] = p[i] * x[i + 1] + q[i]

    return p, q, x


def print_vector(name, vector):
    print(name)
    for i, value in enumerate(vector, start=1):
        print(f"x{i} = {value:.10f}")
    print()


def main():
    a = [0.0, -6.0, 6.0, -7.0, 5.0]
    b = [7.0, 19.0, -18.0, -11.0, -7.0]
    c = [-5.0, -9.0, 7.0, -2.0, 0.0]
    d = [38.0, 14.0, -45.0, 30.0, 48.0]

    p, q, x = thomas_algorithm(a, b, c, d)

    print("Задание 1.2, вариант 23\n")
    print_vector("Прогоночные коэффициенты P:", p)
    print_vector("Прогоночные коэффициенты Q:", q)
    print_vector("Решение системы:", x)


if __name__ == "__main__":
    main()
