EPS = 1e-12


def f(x, y, z):
    return z


def g(x, y, z):
    return 2 * y / (x**2 + 1)


def p(x):
    return 0.0


def q(x):
    return -2 / (x**2 + 1)


def exact_solution(x):
    return x**2 + 1


def exact_derivative(x):
    return 2 * x


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


def get_finite_difference_errors(points):
    result = []

    for x, y in points:
        y_exact = exact_solution(x)
        error = abs(y_exact - y)
        result.append((x, y, y_exact, error))

    return result


def get_max_error(points):
    errors = get_errors(points)
    return max(error for x, y, z, y_exact, error in errors)


def get_max_finite_difference_error(points):
    errors = get_finite_difference_errors(points)
    return max(error for x, y, y_exact, error in errors)


def runge_romberg_error(points_h, points_2h, p, is_system=True):
    if p <= 0:
        raise ValueError("Порядок метода должен быть положительным")

    result = []
    denominator = 2**p - 1

    for i in range(len(points_2h)):
        if is_system:
            x_2h, y_2h, z_2h = points_2h[i]
            x_h, y_h, z_h = points_h[2 * i]
        else:
            x_2h, y_2h = points_2h[i]
            x_h, y_h = points_h[2 * i]

        if abs(x_h - x_2h) > EPS:
            raise ValueError("Сетки с шагами h и 2h не совпадают в общих узлах")

        error = (y_h - y_2h) / denominator
        result.append((x_h, y_h, y_2h, error, abs(error)))

    return result


def tridiagonal_solve(lower, middle, upper, right):
    n = len(middle)

    if not (
        len(lower) == n
        and len(upper) == n
        and len(right) == n
    ):
        raise ValueError("Размеры диагоналей системы не совпадают")

    p_coef = [0.0 for _ in range(n)]
    q_coef = [0.0 for _ in range(n)]
    result = [0.0 for _ in range(n)]

    if abs(middle[0]) < EPS:
        raise ValueError("Нулевой элемент на главной диагонали")

    p_coef[0] = -upper[0] / middle[0]
    q_coef[0] = right[0] / middle[0]

    for i in range(1, n):
        denominator = middle[i] + lower[i] * p_coef[i - 1]

        if abs(denominator) < EPS:
            raise ValueError("Метод прогонки не может продолжаться: деление на ноль")

        p_coef[i] = -upper[i] / denominator if i < n - 1 else 0.0
        q_coef[i] = (right[i] - lower[i] * q_coef[i - 1]) / denominator

    result[n - 1] = q_coef[n - 1]

    for i in range(n - 2, -1, -1):
        result[i] = p_coef[i] * result[i + 1] + q_coef[i]

    return result


def print_shooting_iterations_table(title, iterations):
    print(title)

    z_title = "y'(2)"

    print(
        f"{'j':<4}"
        f"{'eta':>18}"
        f"{'y(2)':>18}"
        f"{z_title:>18}"
        f"{'Phi(eta)':>18}"
    )

    for j, eta, y_b, z_b, phi in iterations:
        print(
            f"{j:<4}"
            f"{eta:>18.10f}"
            f"{y_b:>18.10f}"
            f"{z_b:>18.10f}"
            f"{phi:>18.10f}"
        )

    print()


def print_shooting_table(title, points):
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


def print_finite_difference_table(title, points):
    print(title)

    print(
        f"{'k':<4}"
        f"{'x':>14}"
        f"{'y':>18}"
        f"{'y точное':>18}"
        f"{'погрешность':>18}"
    )

    errors = get_finite_difference_errors(points)

    for k, (x, y, y_exact, error) in enumerate(errors):
        print(
            f"{k:<4}"
            f"{x:>14.10f}"
            f"{y:>18.10f}"
            f"{y_exact:>18.10f}"
            f"{error:>18.10f}"
        )

    print(
        "Максимальная погрешность: "
        f"{get_max_finite_difference_error(points):.10f}"
    )
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