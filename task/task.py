from utils import (
    A_COEF,
    T,
    initial_grid,
    tridiagonal_solve,
    compute_errors,
    build_plots,
    run_grid_convergence_study,
)


def solve_explicit(n=30, k_total=1200, a=A_COEF, t_max=T):
    x_values, t_values, u, h, tau, sigma = initial_grid(n, k_total, a, t_max)

    # Условие устойчивости Куранта для явной схемы параболического типа: sigma <= 0.5
    # Если sigma > 0.5, схема теряет устойчивость и решение экспоненциально "разваливается"
    if sigma > 0.5 + 1e-12:
        raise ValueError(f"Явная схема неустойчива: sigma = {sigma:.4f} > 0.5")

    layers = [list(u)]

    for _ in range(k_total):
        u_next = [0.0] * (n + 1)
        # Внутренние узлы j от 1 до n-1 вычисляются по трехточечному шаблону
        for j in range(1, n):
            u_next[j] = u[j] + sigma * (u[j - 1] - 2.0 * u[j] + u[j + 1])

        # Граничные условия 1-го рода: u(0, t) = 0 и u(1, t) = 0.
        # Производных на границах нет, поэтому зануляем значения на краях напрямую
        # без дополнительных аппроксимаций 1-го или 2-го порядков
        u_next[0] = 0.0
        u_next[n] = 0.0

        u = u_next
        layers.append(list(u))

    return x_values, t_values, layers, sigma


def solve_weighted(n=30, k_total=1200, a=A_COEF, t_max=T, theta=1.0):
    x_values, t_values, u, h, tau, sigma = initial_grid(n, k_total, a, t_max)
    m = n - 1  # Число внутренних неизвестных узлов (j = 1 ... n-1)

    # Коэффициенты трехдиагональной матрицы СЛАУ на новом временном слое (k + 1):
    # -theta*sigma * u_{j-1} + (1 + 2*theta*sigma) * u_j - theta*sigma * u_{j+1} = right_j
    lower = [-theta * sigma] * m
    middle = [1.0 + 2.0 * theta * sigma] * m
    upper = [-theta * sigma] * m

    layers = [list(u)]

    for _ in range(k_total):
        right = [0.0] * m
        # Формируем правую часть из значений предыдущего временного слоя (явная часть)
        for i in range(m):
            j = i + 1
            expl = (1.0 - theta) * sigma * (u[j - 1] - 2.0 * u[j] + u[j + 1])
            right[i] = u[j] + expl

        # Так как u_0^{k+1} = 0 и u_N^{k+1} = 0, граничные члены в уравнениях
        # для j=1 и j=N-1 равны нулю и не требуют переноса в правую часть

        # Решаем СЛАУ с трехдиагональной матрицей методом скалярной прогонки
        internal = tridiagonal_solve(lower, middle, upper, right)

        # Добавляем краевые условия 1-го рода (нулевые граничные значения)
        u = [0.0] + internal + [0.0]
        layers.append(list(u))

    return x_values, t_values, layers, sigma


def solve_implicit(n=30, k_total=1200, a=A_COEF, t_max=T):
    # Чисто неявная схема: theta = 1.0.
    # Абсолютно устойчива при любых шагах h и tau, порядок аппроксимации O(tau + h^2)
    return solve_weighted(n, k_total, a, t_max, theta=1.0)


def solve_crank_nicolson(n=30, k_total=1200, a=A_COEF, t_max=T):
    # Схема Кранка-Николсона: полунеявная схема с симметричным весом theta = 0.5.
    # Абсолютно устойчива, имеет повышенный порядок аппроксимации O(tau^2 + h^2)
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

    # Построение базовых графиков решения
    build_plots(solutions, x_base, t_base)

    # Исследование зависимости погрешности от параметров сетки h и tau
    run_grid_convergence_study(schemes)


if __name__ == "__main__":
    main()
