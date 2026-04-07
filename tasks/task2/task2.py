from utils import EPS, print_vector, check_solution


def forward_solve(a, b, c, d):
    n = len(d)
    p = [0.0] * n
    q = [0.0] * n

    p[0] = -c[0] / b[0]
    q[0] = d[0] / b[0]

    for i in range(1, n - 1):
        denom = a[i] * p[i-1] + b[i]
        if abs(denom) < EPS:
            raise ValueError(f"Нулевой знаменатель на шаге {i}")
        p[i] = -c[i] / denom
        q[i] = (d[i] - a[i] * q[i-1]) / denom

    p[n-1] = 0.0
    denom = a[n-1] * p[n-2] + b[n-1]
    if abs(denom) < EPS:
        raise ValueError("Нулевой знаменатель на последнем шаге")
    q[n-1] = (d[n-1] - a[n-1] * q[n-2]) / denom

    return p, q


def backward_solve(p, q):
    n = len(q)
    x = [0.0] * n
    x[n-1] = q[n-1]

    for i in range(n-2, -1, -1):
        x[i] = p[i] * x[i+1] + q[i]

    return x


def main():
    a = [0.0, -6.0, 6.0, -7.0, 5.0]
    b = [7.0, 19.0, -18.0, -11.0, -7.0]
    c = [-5.0, -9.0, 7.0, -2.0, 0.0]
    d = [38.0, 14.0, -45.0, 30.0, 48.0]

    p, q = forward_solve(a, b, c, d)
    x = backward_solve(p, q)

    print_vector("Прогоночные коэффициенты P:", p, "p")
    print_vector("Прогоночные коэффициенты Q:", q, "q")
    print_vector("Решение системы:", x, "x")
    check_solution(a, b, c, d, x)


if __name__ == "__main__":
    main()
