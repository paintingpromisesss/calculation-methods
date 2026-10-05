import math


EPS = 1e-12

# Параметры задачи (вариант 10): a = b = mu = 1, область [0, pi] x [0, pi]
A = 1.0
B = 1.0
MU = 1.0
L_X = math.pi
L_Y = math.pi
T = 1.0


def exact_solution(x, y, t):
    # Аналитическое решение U(x, y, t) = sin x sin y sin(mu t)
    return math.sin(x) * math.sin(y) * math.sin(MU * t)


def source(x, y, t):
    # Источник f = sin x sin y (cos(mu t) + (a + b) sin(mu t))
    return (
        math.sin(x)
        * math.sin(y)
        * (math.cos(MU * t) + (A + B) * math.sin(MU * t))
    )


def get_grid(l, h):
    # Равномерная сетка на [0, l] с шагом h
    if h <= 0:
        raise ValueError("Шаг должен быть положительным")

    n = int(round(l / h))

    if n < 3:
        raise ValueError("Сетка должна содержать минимум 3 внутренних узла")

    if abs(l / h - n) > EPS:
        raise ValueError("Длина области не делится на шаг нацело")

    return [i * h for i in range(n + 1)]


def tridiagonal_solve(lower, middle, upper, right):
    # Метод прогонки для трехдиагональной системы
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


def set_x_boundary(u, x_values, y_values, t):
    # Границы по x: u(0, y, t) = 0 (первый род),
    # u_x(pi, y, t) = -sin y sin(mu t) (второй род).
    # Производная аппроксимируется двухточечно: (u_I - u_{I-1}) / h_x.
    i_max = len(x_values) - 1
    h_x = x_values[1]

    for j in range(len(y_values)):
        u[0][j] = 0.0
        u[i_max][j] = u[i_max - 1][j] - h_x * math.sin(y_values[j]) * math.sin(MU * t)


def set_y_boundary(u, x_values, y_values, t):
    # Границы по y: u(x, 0, t) = 0, u_y(x, pi, t) = -sin x sin(mu t)
    j_max = len(y_values) - 1
    h_y = y_values[1]

    for i in range(len(x_values)):
        u[i][0] = 0.0
        u[i][j_max] = u[i][j_max - 1] - h_y * math.sin(x_values[i]) * math.sin(MU * t)


def get_max_error(u, x_values, y_values, t):
    # Максимальная по сетке погрешность в момент t
    error = 0.0

    for i in range(len(x_values)):
        for j in range(len(y_values)):
            difference = abs(u[i][j] - exact_solution(x_values[i], y_values[j], t))

            if difference > error:
                error = difference

    return error


def print_solution_table(title, u, x_values, y_values, t):
    # Таблица значений вдоль диагонали x = y с погрешностью
    print(title)
    print(f"t = {t:.2f}")
    print()

    print(
        f"{'i':<4}"
        f"{'x':>14}"
        f"{'y':>14}"
        f"{'u числ.':>18}"
        f"{'u точное':>18}"
        f"{'погрешность':>18}"
    )

    count = min(len(x_values), len(y_values))

    for i in range(count):
        x = x_values[i]
        y = y_values[i]
        u_exact = exact_solution(x, y, t)
        error = abs(u[i][i] - u_exact)

        print(
            f"{i:<4}"
            f"{x:>14.6f}"
            f"{y:>14.6f}"
            f"{u[i][i]:>18.10f}"
            f"{u_exact:>18.10f}"
            f"{error:>18.10f}"
        )

    print(
        "Максимальная погрешность по всей сетке: "
        f"{get_max_error(u, x_values, y_values, t):.10f}"
    )
    print()


def print_research_table(title, rows):
    # Таблица исследования: rows = (шаг, погрешность, отношение погрешностей)
    print(title)

    print(
        f"{'шаг':>14}"
        f"{'погрешность':>18}"
        f"{'отношение':>14}"
    )

    previous = None

    for step, error in rows:
        ratio = error / previous if previous else None

        if ratio is None:
            print(f"{step:>14.6f}{error:>18.10f}{'':>14}")
        else:
            print(f"{step:>14.6f}{error:>18.10f}{ratio:>14.6f}")

        previous = error

    print()
