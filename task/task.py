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


# Явная схема «крест»
def solve_explicit(n=40, k_total=100, a=A_COEF, t_max=T, init_order=2):
    x_values, t_values, u_0, u_1, _, tau, sigma = initial_grid(
        n, k_total, a, t_max, init_order=init_order
    )

    # Условие устойчивости Куранта: sigma = (a*tau/h)^2 <= 1
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

        for j in range(1, n):
            u_next[j] = (
                2.0 * u_cur[j]
                - u_prev[j]
                + sigma * (u_cur[j - 1] - 2.0 * u_cur[j] + u_cur[j + 1])
            )

        # Условия Дирихле задаются напрямую, без аппроксимации производных
        u_next[0] = -math.sin(a * t_next)
        u_next[n] = math.sin(a * t_next)

        u_prev = u_cur
        u_cur = u_next
        layers.append(list(u_cur))

    return x_values, t_values, layers, sigma


# Неявная схема: безусловно устойчива
def solve_implicit(n=40, k_total=100, a=A_COEF, t_max=T, init_order=2):
    x_values, t_values, u_0, u_1, _, tau, sigma = initial_grid(
        n, k_total, a, t_max, init_order=init_order
    )
    m = n - 1

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

        right = [2.0 * u_cur[j] - u_prev[j] for j in range(1, n)]

        right[0] += sigma * u_next_0
        right[-1] += sigma * u_next_n

        internal = tridiagonal_solve(lower, middle, upper, right)

        u_next = [u_next_0] + internal + [u_next_n]

        u_prev = u_cur
        u_cur = u_next
        layers.append(list(u_cur))

    return x_values, t_values, layers, sigma


def solve_explicit_order1(n=40, k_total=100, a=A_COEF, t_max=T):
    return solve_explicit(n, k_total, a, t_max, init_order=1)


def solve_implicit_order1(n=40, k_total=100, a=A_COEF, t_max=T):
    return solve_implicit(n, k_total, a, t_max, init_order=1)


def main():
    n = 40
    k_total = 100

    schemes = [
        ("Явная (1-й пор. НУ)", solve_explicit_order1),
        ("Неявная (1-й пор. НУ)", solve_implicit_order1),
        ("Явная (2-й пор. НУ)", solve_explicit),
        ("Неявная (2-й пор. НУ)", solve_implicit),

    ]

    h = L / n
    tau = T / k_total
    sigma = (A_COEF * tau / h) ** 2

    print(f"Базовый расчет: N={n}, K={k_total}, h={h:.5f}, tau={tau:.6f}, sigma={sigma:.4f}")
    print(f"{'Метод':<24}{f'max|ошибка| при T={T:g}':<24}{'RMSE (0..T)'}")

    solutions = {}
    x_base = None
    t_base = None

    for name, solver in schemes:
        x, t, layers, _ = solver(n, k_total)
        solutions[name] = layers
        x_base, t_base = x, t

        max_err, rmse = compute_errors(layers, x, t)
        print(f"{name:<24}{max_err:<24.6e}{rmse:.6e}")

    assert x_base is not None and t_base is not None, "Не выбраны схемы для расчета"

    method_order = [
        "Явная (1-й пор. НУ)",
        "Явная (2-й пор. НУ)",
        "Неявная (1-й пор. НУ)",
        "Неявная (2-й пор. НУ)",
    ]

    method_labels = [
        "Явная НУ-1",
        "Явная НУ-2",
        "Неявная НУ-1",
        "Неявная НУ-2",
    ]

    time_points = (0.0, 0.25, 0.5, 0.75, 1.0)

    for metric_name, metric_idx in [
        ("Максимальная абсолютная погрешность", 0),
        ("Среднеквадратичная погрешность (RMSE)", 1),
    ]:
        print(f"\n{metric_name}")
        print("-" * 85)

        print(f"{'t':<10}", end="")
        for label in method_labels:
            print(f"{label:>18}", end="")
        print()

        print("-" * 85)

        for fraction in time_points:
            k = round(fraction * k_total)
            t = t_base[k]

            print(f"{t:<10.2f}", end="")

            for name in method_order:
                layers = solutions[name]
                errors = compute_errors(
                    [layers[k]], x_base, [t]
                )
                print(f"{errors[metric_idx]:>18.3e}", end="")

            print()

    build_plots(solutions, x_base, t_base)

    run_grid_convergence_study(schemes)


if __name__ == "__main__":
    main()
