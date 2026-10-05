import math


# Параметры задачи варианта 10:
# u_tt + 3 u_t = u_xx + u_x - u - cos(x) exp(-t),  0 < x < pi,  t > 0
# u_x(0, t) = exp(-t),  u_x(pi, t) = -exp(-t)
# u(x, 0) = sin(x),  u_t(x, 0) = -sin(x)
# Аналитическое решение: U(x, t) = exp(-t) sin(x).
# Приводим к виду u_tt = a^2 u_xx + b u_x + c u - d u_t + f(x, t):
# a^2 = 1, b = 1, c = -1, d = 3, f = -cos(x) exp(-t).

A_SQ = 1.0
B = 1.0
C = -1.0
D = 3.0

L = math.pi
T = 1.0


def f(x, t):
    return -math.cos(x) * math.exp(-t)


def psi1(x):
    return math.sin(x)


def psi2(x):
    return -math.sin(x)


def phi_left(t):
    return math.exp(-t)


def phi_right(t):
    return -math.exp(-t)


def exact_solution(x, t):
    return math.exp(-t) * math.sin(x)


def check_source_consistency(n_x=21, n_t=21):
    # Численная проверка согласованности источника f и аналитического решения:
    # невязка r = (u_tt + 3u_t) - (u_xx + u_x - u + f) на U = exp(-t) sin(x).
    max_residual = 0.0

    for i in range(n_x):
        x = L * i / (n_x - 1)

        for k in range(n_t):
            t = T * k / (n_t - 1)
            e = math.exp(-t)
            s = math.sin(x)
            cx = math.cos(x)

            utt = e * s
            ut = -e * s
            uxx = -e * s
            ux = e * cx
            u = e * s

            residual = utt + D * ut - (uxx + ux - u + f(x, t))
            max_residual = max(max_residual, abs(residual))

    return max_residual


def get_grid_x(n_x):
    if n_x < 2:
        raise ValueError("Число разбиений по x должно быть не меньше 2")

    h = L / n_x
    return [i * h for i in range(n_x + 1)]


def get_time_steps(tau):
    if tau <= 0:
        raise ValueError("Шаг по времени должен быть положительным")

    n_t = int(round(T / tau))

    if n_t < 2:
        raise ValueError("Шаг по времени слишком велик")

    return n_t, T / n_t


def get_layer_errors(u, x_values, t):
    # Погрешности во всех узлах одного временного слоя.
    if len(u) != len(x_values):
        raise ValueError("Число узлов сетки и значений решения не совпадает")

    result = []

    for j in range(len(x_values)):
        y_exact = exact_solution(x_values[j], t)
        result.append((x_values[j], u[j], y_exact, abs(u[j] - y_exact)))

    return result


def get_max_error(u, x_values, t):
    errors = get_layer_errors(u, x_values, t)
    return max(error for _, _, _, error in errors)


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

    if abs(middle[0]) < 1e-14:
        raise ValueError("Нулевой элемент на главной диагонали")

    p_coef[0] = -upper[0] / middle[0]
    q_coef[0] = right[0] / middle[0]

    for i in range(1, n):
        denominator = middle[i] + lower[i] * p_coef[i - 1]

        if abs(denominator) < 1e-14:
            raise ValueError("Метод прогонки не может продолжаться: деление на ноль")

        p_coef[i] = -upper[i] / denominator if i < n - 1 else 0.0
        q_coef[i] = (right[i] - lower[i] * q_coef[i - 1]) / denominator

    result[n - 1] = q_coef[n - 1]

    for i in range(n - 2, -1, -1):
        result[i] = p_coef[i] * result[i + 1] + q_coef[i]

    return result


def print_layer_table(title, u, x_values, t, step=5):
    # Таблица погрешности на одном временном слое (каждый step-й узел).
    print(title)
    print(f"t = {t:.10f}")

    print(
        f"{'j':<4}"
        f"{'x':>14}"
        f"{'u':>18}"
        f"{'u точное':>18}"
        f"{'погрешность':>18}"
    )

    for j, (x, y, y_exact, error) in enumerate(
        get_layer_errors(u, x_values, t)
    ):
        if j % step == 0 or j == len(x_values) - 1:
            print(
                f"{j:<4}"
                f"{x:>14.10f}"
                f"{y:>18.10f}"
                f"{y_exact:>18.10f}"
                f"{error:>18.10e}"
            )

    print()


def print_error_summary(title, times, errors):
    print(title)

    print(
        f"{'t':>14}"
        f"{'погрешность':>18}"
    )

    for t, error in zip(times, errors):
        print(
            f"{t:>14.4f}"
            f"{error:>18.10e}"
        )

    print()


def print_convergence_table(title, rows):
    # rows: (h, tau, error, ratio) - погрешность при t = T.
    print(title)

    print(
        f"{'h':>14}"
        f"{'tau':>14}"
        f"{'E(T)':>18}"
        f"{'E(2h)/E(h)':>12}"
    )

    for h, tau, error, ratio in rows:
        ratio_text = f"{ratio:.2f}" if ratio is not None else "-"

        print(
            f"{h:>14.6f}"
            f"{tau:>14.6f}"
            f"{error:>18.10e}"
            f"{ratio_text:>12}"
        )

    print()


def print_orders_table(title, rows):
    # rows: (scheme, init_order, boundary_name, order).
    print(title)

    print(
        f"{'схема':<10}"
        f"{'нач.усл.':>10}"
        f"{'гранусл.':>26}"
        f"{'порядок':>10}"
    )

    for scheme_name, init_order, boundary_name, order in rows:
        print(
            f"{scheme_name:<10}"
            f"{init_order:>10}"
            f"{boundary_name:>26}"
            f"{order:>10.2f}"
        )

    print()
