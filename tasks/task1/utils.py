EPS = 1e-12


def f(x, y, z):
    if abs(x) < EPS:
        raise ValueError("x не должен быть равен нулю")

    return z


def g(x, y, z):
    if abs(x) < EPS:
        raise ValueError("x не должен быть равен нулю")

    return (-x * z + y + 3 * x**2) / x**2


def exact_solution(x):
    if abs(x) < EPS:
        raise ValueError("x не должен быть равен нулю")

    return x**2 + x + 1 / x


def exact_derivative(x):
    if abs(x) < EPS:
        raise ValueError("x не должен быть равен нулю")

    return 2 * x + 1 - 1 / x**2


def get_n(a, b, h):
    if h <= 0:
        raise ValueError("Шаг должен быть положительным")

    return int(round((b - a) / h))


def get_grid(a, b, h):
    n = get_n(a, b, h)
    return [a + i * h for i in range(n + 1)]


def get_errors(points):
    result = []

    for x, y, z in points:
        y_exact = exact_solution(x)
        error = abs(y_exact - y)
        result.append((x, y, z, y_exact, error))

    return result


def get_max_error(points):
    errors = get_errors(points)
    return max(error for x, y, z, y_exact, error in errors)


def runge_romberg_error(points_h, points_2h, p):
    if p <= 0:
        raise ValueError("Порядок метода должен быть положительным")

    result = []
    denominator = 2**p - 1

    for i in range(len(points_2h)):
        x_2h, y_2h, z_2h = points_2h[i]
        x_h, y_h, z_h = points_h[2 * i]

        if abs(x_h - x_2h) > EPS:
            raise ValueError("Сетки с шагами h и 2h не совпадают в общих узлах")

        error = (y_h - y_2h) / denominator
        result.append((x_h, y_h, y_2h, error, abs(error)))

    return result


def print_method_table(title, points):
    print(title)

    print(
        f"{'k':<4}"
        f"{'x':>14}"
        f"{'y':>18}"
        f"{'z':>18}"
        f"{'y точное':>18}"
        f"{'погрешность':>18}"
    )

    errors = get_errors(points)

    for k, (x, y, z, y_exact, error) in enumerate(errors):
        print(
            f"{k:<4}"
            f"{x:>14.10f}"
            f"{y:>18.10f}"
            f"{z:>18.10f}"
            f"{y_exact:>18.10f}"
            f"{error:>18.10f}"
        )

    print(f"Максимальная погрешность: {get_max_error(points):.10f}")
    print()


def print_runge_romberg_table(title, errors):
    print(title)

    print(
        f"{'k':<4}"
        f"{'x':>14}"
        f"{'y(h)':>18}"
        f"{'y(2h)':>18}"
        f"{'R':>18}"
        f"{'|R|':>18}"
    )

    for k, (x, y_h, y_2h, error, abs_error) in enumerate(errors):
        print(
            f"{k:<4}"
            f"{x:>14.10f}"
            f"{y_h:>18.10f}"
            f"{y_2h:>18.10f}"
            f"{error:>18.10f}"
            f"{abs_error:>18.10f}"
        )

    print()