import math

EPS = 1e-12

# Параметры задачи (вариант 10): u_t = a^2 u_xx + b u_x + c u
A_COEF = 1.0
B_COEF = 1.0
C_COEF = -1.0
L = math.pi
T = 1.0


def exact_solution(x, t):
    """Аналитическое решение U(x,t) = exp((c-a)t) sin(x + bt)."""
    return math.exp((C_COEF - A_COEF) * t) * math.sin(x + B_COEF * t)


def phi_left(t):
    """Граничное условие при x=0: u_x + u = exp((c-a)t)(cos(bt)+sin(bt))."""
    e = math.exp((C_COEF - A_COEF) * t)
    return e * (math.cos(B_COEF * t) + math.sin(B_COEF * t))


def phi_right(t):
    """Граничное условие при x=pi: u_x + u = -exp((c-a)t)(cos(bt)+sin(bt))."""
    return -phi_left(t)


def get_grid_x(n):
    if n < 3:
        raise ValueError("Нужно минимум 3 узла по x")
    if n < 0:
        raise ValueError("Число разбиений должно быть неотрицательным")
    h = L / n
    return [j * h for j in range(n + 1)]


def get_tau(k_total):
    if k_total < 1:
        raise ValueError("Число шагов по времени должно быть не меньше 1")
    return T / k_total


def get_max_error(u_layer, x_values, t):
    """Максимальная по сетке погрешность в момент времени t."""
    if len(u_layer) != len(x_values):
        raise ValueError("Размеры слоя решения и сетки не совпадают")
    return max(
        abs(u_layer[j] - exact_solution(x_values[j], t))
        for j in range(len(x_values))
    )


def tridiagonal_solve(lower, middle, upper, right):
    """Метод прогонки для системы с трехдиагональной матрицей."""
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


def print_solution_table(title, x_values, u_layer, t):
    """Таблица численного решения в момент времени t с точным решением."""
    print(title)
    print(f"{'j':<4}{'x':>14}{'u(x, t)':>18}{'u точное':>18}{'погрешность':>18}")

    for j in range(len(x_values)):
        u_exact = exact_solution(x_values[j], t)
        error = abs(u_layer[j] - u_exact)
        print(
            f"{j:<4}"
            f"{x_values[j]:>14.8f}"
            f"{u_layer[j]:>18.10f}"
            f"{u_exact:>18.10f}"
            f"{error:>18.10f}"
        )

    print(f"Максимальная погрешность: {get_max_error(u_layer, x_values, t):.10f}")
    print()


def print_convergence_table(title, n_values, k_values, errors):
    """Таблица исследования погрешности от сеточных параметров.

    errors[i][j] - максимальная погрешность на сетке (n_values[i], k_values[j]).
    """
    print(title)

    print("N \\ K", end="")
    for k in k_values:
        print(f"{k:>12}", end="")
    print()


def print_grid_error_table(title, grid_params, errors):
    """Таблица погрешности на последовательности сеток.

    grid_params - список (N, K), errors - соответствующие максимальные
    погрешности в момент времени T.
    """
    print(title)

    print(f"{'N':>6}{'K':>8}{'h':>14}{'tau':>14}{'max погрешность':>18}")

    for (n, k), error in zip(grid_params, errors):
        print(
            f"{n:>6}"
            f"{k:>8}"
            f"{L / n:>14.6f}"
            f"{T / k:>14.6f}"
            f"{error:>18.10f}"
        )

    print()


def print_order_table(title, grid_params, errors, ref_order):
    """Таблица порядков сходимости при измельчении сетки.

    Порядок оценивается по отношению погрешностей на соседних сетках
    с одновременным измельчением h и tau: p = log2(e_h / e_{h/2}).
    """
    print(title)

    print(f"{'N':>6}{'K':>8}{'max погрешность':>18}{'порядок':>12}")

    for i, ((n, k), error) in enumerate(zip(grid_params, errors)):
        if i == 0:
            print(f"{n:>6}{k:>8}{error:>18.10f}{'-':>12}")
        else:
            order = math.log(errors[i - 1] / error, 2)
            print(
                f"{n:>6}"
                f"{k:>8}"
                f"{error:>18.10f}"
                f"{order:>12.2f}"
            )

    print(
        f"Теоретический порядок: {ref_order}"
    )
    print()
