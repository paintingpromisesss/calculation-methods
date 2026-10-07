from utils import (
    A_COEF,
    T,
    initial_grid,
    tridiagonal_solve,
    compute_errors,
    build_plots,
    run_grid_convergence_study,
)

# Явная
def solve_explicit(n=30, k_total=1200, a=A_COEF, t_max=T):
    x_values, t_values, u, sigma = initial_grid(n, k_total, a, t_max)

    # Условие устойчивости Куранта для явной схемы параболического типа: sigma <= 0.5
    # Если sigma > 0.5, схема теряет устойчивость и решение экспоненциально "разваливается"
    if sigma > 0.5 + 1e-12:
        raise ValueError(f"Явная схема неустойчива: sigma = {sigma:.4f} > 0.5")

    layers = [list(u)]

    for _ in range(k_total):
        u_next = [0.0] * (n + 1)
        for j in range(1, n):
            u_next[j] = u[j] + sigma * (u[j - 1] - 2.0 * u[j] + u[j + 1])

        # Граничные условия 1-го рода: u(0, t) = 0 и u(1, t) = 0.
        # Производных на границах нет, поэтому зануляем значения на краях напрямую
        u_next[0] = 0.0
        u_next[n] = 0.0

        u = u_next
        layers.append(list(u))

    return x_values, t_values, layers, sigma


def solve_weighted(n=30, k_total=1200, a=A_COEF, t_max=T, theta=1.0):
    x_values, t_values, u, sigma = initial_grid(n, k_total, a, t_max)
    m = n - 1

    # Коэффициенты трехдиагональной матрицы СЛАУ на новом временном слое (k + 1)
    lower = [-theta * sigma] * m
    middle = [1.0 + 2.0 * theta * sigma] * m
    upper = [-theta * sigma] * m

    layers = [list(u)]

    for _ in range(k_total):
        right = [0.0] * m
        for i in range(m):
            j = i + 1
            expl = (1.0 - theta) * sigma * (u[j - 1] - 2.0 * u[j] + u[j + 1])
            right[i] = u[j] + expl

        internal = tridiagonal_solve(lower, middle, upper, right)

        u = [0.0] + internal + [0.0]
        layers.append(list(u))

    return x_values, t_values, layers, sigma

# Неявная
def solve_implicit(n=30, k_total=1200, a=A_COEF, t_max=T):
    return solve_weighted(n, k_total, a, t_max, theta=1.0)


def solve_crank_nicolson(n=30, k_total=1200, a=A_COEF, t_max=T):
    return solve_weighted(n, k_total, a, t_max, theta=0.5)


def main():
    n = 30
    k_total = 1200

    schemes = [
        ("Явная", solve_explicit),
        ("Неявная", solve_implicit),
        ("Кранк–Николсон", solve_crank_nicolson),
    ]

    h = 1.0 / n
    tau = T / k_total
    sigma = A_COEF * tau / (h**2)

    print(f"Базовый расчет: N={n}, K={k_total}, h={h:.5f}, tau={tau:.6f}, sigma={sigma:.4f}")
    print(f"{'Метод':<22}{'max|ошибка| при T':<24}{'RMSE'}")

    solutions = {}
    x_base = None
    t_base = None

    for name, solver in schemes:
        x, t, layers, _ = solver(n, k_total)
        solutions[name] = layers
        x_base, t_base = x, t

        max_err, rmse = compute_errors(layers, x, t)
        print(f"{name:<22}{max_err:<24.6e}{rmse:.6e}")

    build_plots(solutions, x_base, t_base)

    run_grid_convergence_study(schemes)


if __name__ == "__main__":
    main()
