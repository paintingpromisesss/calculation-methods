import math
from utils import (
    A_COEF,
    L,
    T,
    initial_grid,
    tridiagonal_solve,
    compute_errors,
    build_plots,
    run_grid_convergence_study,
)


def solve_explicit(n=40, k_total=100, a=A_COEF, t_max=T, init_order=2):
    x_values, t_values, u_0, u_1, h, tau, sigma = initial_grid(
        n, k_total, a, t_max, init_order=init_order
    )

    # Условие устойчивости Куранта-Фридрихса-Леви (CFL) для явной схемы "крест":
    # sigma = (a * tau / h)^2 <= 1.0. При sigma > 1.0 схема теряет устойчивость
    if sigma > 1.0 + 1e-12:
        raise ValueError(
            f"Явная схема неустойчива: sigma = (a*tau/h)^2 = {sigma:.4f} > 1.0 (нарушено условие Куранта)"
        )

    layers = [list(u_0), list(u_1)]
    u_prev = list(u_0)
    u_cur = list(u_1)

    for k in range(1, k_total):
        t_next = (k + 1) * tau
        u_next = [0.0] * (n + 1)

        # Внутренние узлы j от 1 до n-1 вычисляются по схеме "крест":
        # (u_j^{k+1} - 2*u_j^k + u_j^{k-1}) / tau^2 = a^2 * (u_{j-1}^k - 2*u_j^k + u_{j+1}^k) / h^2
        for j in range(1, n):
            u_next[j] = (
                2.0 * u_cur[j]
                - u_prev[j]
                + sigma * (u_cur[j - 1] - 2.0 * u_cur[j] + u_cur[j + 1])
            )

        # Граничные условия 1-го рода (Дирихле): u(0, t) = -sin(a*t) и u(pi, t) = sin(a*t).
        # Производных на границах нет, поэтому задаем точные значения на краях напрямую
        # без дополнительных аппроксимаций 1-го или 2-го порядков
        u_next[0] = -math.sin(a * t_next)
        u_next[n] = math.sin(a * t_next)

        u_prev = u_cur
        u_cur = u_next
        layers.append(list(u_cur))

    return x_values, t_values, layers, sigma


def solve_implicit(n=40, k_total=100, a=A_COEF, t_max=T, init_order=2):
    x_values, t_values, u_0, u_1, h, tau, sigma = initial_grid(
        n, k_total, a, t_max, init_order=init_order
    )
    m = n - 1  # Число внутренних неизвестных узлов (j = 1 ... n-1)

    # Коэффициенты трехдиагональной матрицы СЛАУ на новом временном слое (k + 1):
    # -sigma * u_{j-1}^{k+1} + (1 + 2*sigma) * u_j^{k+1} - sigma * u_{j+1}^{k+1} = 2*u_j^k - u_j^{k-1}
    # Схема абсолютно устойчива при любых шагах h и tau благодаря диагональному преобладанию:
    # |1 + 2*sigma| > 2*sigma.
    lower = [-sigma] * m
    middle = [1.0 + 2.0 * sigma] * m
    upper = [-sigma] * m

    layers = [list(u_0), list(u_1)]
    u_prev = list(u_0)
    u_cur = list(u_1)

    for k in range(1, k_total):
        t_next = (k + 1) * tau
        u_next_0 = -math.sin(a * t_next)
        u_next_n = math.sin(a * t_next)

        # Формируем правую часть из значений двух предыдущих временных слоев
        right = [2.0 * u_cur[j] - u_prev[j] for j in range(1, n)]

        # Учитываем граничные условия Дирихле на краях отрезка для j=1 и j=n-1
        right[0] += sigma * u_next_0
        right[-1] += sigma * u_next_n

        # Решаем СЛАУ с трехдиагональной матрицей методом скалярной прогонки
        internal = tridiagonal_solve(lower, middle, upper, right)

        # Собираем полный временной слой с краевыми условиями Дирихле
        u_next = [u_next_0] + internal + [u_next_n]

        u_prev = u_cur
        u_cur = u_next
        layers.append(list(u_cur))

    return x_values, t_values, layers, sigma


def solve_explicit_order1(n=40, k_total=100, a=A_COEF, t_max=T):
    # Явная схема крест с аппроксимацией 2-го начального условия 1-м порядком:
    # u_j^1 = u_j^0 + tau * psi2(x_j) (погрешность O(tau))
    return solve_explicit(n, k_total, a, t_max, init_order=1)


def solve_implicit_order1(n=40, k_total=100, a=A_COEF, t_max=T):
    # Неявная схема с аппроксимацией 2-го начального условия 1-м порядком
    return solve_implicit(n, k_total, a, t_max, init_order=1)


def main():
    n = 40
    k_total = 100

    schemes = [
        ("Явная (крест)", solve_explicit),
        ("Неявная", solve_implicit),
        ("Явная (1-й пор. НУ)", solve_explicit_order1),
    ]

    h = L / n
    tau = T / k_total
    sigma = (A_COEF * tau / h) ** 2

    print(f"Базовый расчет: N={n}, K={k_total}, h={h:.5f}, tau={tau:.6f}, sigma={sigma:.4f}")
    print(f"{'Метод':<24}{'max|ошибка| при T':<24}{'RMSE'}")

    solutions = {}
    x_base = None
    t_base = None

    for name, solver in schemes:
        x, t, layers, _ = solver(n, k_total)
        solutions[name] = layers
        x_base, t_base = x, t

        max_err, rmse = compute_errors(layers, x, t)
        print(f"{name:<24}{max_err:<24.6e}{rmse:.6e}")

    # Построение базовых графиков решения
    build_plots(solutions, x_base, t_base)

    # Исследование зависимости погрешности от параметров сетки h и tau
    run_grid_convergence_study(schemes)


if __name__ == "__main__":
    main()
