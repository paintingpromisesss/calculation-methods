import math
from utils import (
    L_X,
    L_Y,
    initial_grid,
    compute_errors,
    build_plots,
    plot_iterations,
    run_grid_convergence_study,
)


def solve_iterative(n_x=30, n_y=30, eps=1e-8, max_iter=100000,
                    method="liebmann", omega=1.0):
    if not math.isfinite(eps) or eps <= 0.0 or max_iter < 1:
        raise ValueError("Точность и предельное число итераций должны быть положительными")
    if not 0.0 < omega < 2.0:
        raise ValueError("Параметр релаксации должен удовлетворять 0 < omega < 2")

    x_values, y_values, u, h_x, h_y = initial_grid(n_x, n_y)
    coef_x = 1.0 / h_x**2
    coef_y = 1.0 / h_y**2
    denom = 2.0 * (coef_x + coef_y)
    history = []

    for _ in range(max_iter):
        u_next = [list(row) for row in u] if method == "liebmann" else u
        max_diff = 0.0
        for j in range(1, n_y):
            for i in range(1, n_x):
                old_value = u[j][i]
                value = (
                    coef_x * (u[j][i - 1] + u[j][i + 1])
                    + coef_y * (u[j - 1][i] + u[j + 1][i])
                ) / denom
                if method == "relaxation":
                    value = old_value + omega * (value - old_value)
                u_next[j][i] = value
                max_diff = max(max_diff, abs(value - old_value))

        u = u_next
        history.append(max_diff)
        if max_diff <= eps:
            return x_values, y_values, u, history

    raise RuntimeError(f"Метод {method} не сошелся за {max_iter} итераций")


def solve_liebmann(n_x=30, n_y=30, eps=1e-8, max_iter=100000):
    return solve_iterative(n_x, n_y, eps, max_iter, method="liebmann")


def solve_seidel(n_x=30, n_y=30, eps=1e-8, max_iter=100000):
    return solve_iterative(n_x, n_y, eps, max_iter, method="seidel")


def solve_relaxation(n_x=30, n_y=30, eps=1e-8, max_iter=100000, omega=1.7):
    return solve_iterative(n_x, n_y, eps, max_iter, method="relaxation", omega=omega)


def main():
    n_x = 30
    n_y = 30
    eps = 1e-8
    schemes = [
        ("Либман", solve_liebmann),
        ("Зейдель", solve_seidel),
        ("Верхняя релаксация (ω=1.7)", solve_relaxation),
    ]

    print("Вариант 1: u_xx + u_yy = 0, U(x, y) = x + y")
    print(f"Базовый расчет: Nx={n_x}, Ny={n_y}, hx={L_X/n_x:.5f}, hy={L_Y/n_y:.5f}, eps={eps:g}")
    print(f"{'Метод':<30}{'Итерации':<12}{'max|ошибка|':<18}{'RMSE'}")
    solutions = {}
    histories = {}
    for name, solver in schemes:
        x, y, u, history = solver(n_x, n_y, eps)
        solutions[name] = u
        histories[name] = history
        max_err, rmse = compute_errors(u, x, y)
        print(f"{name:<30}{len(history):<12}{max_err:<18.6e}{rmse:.6e}")

    build_plots(solutions, x, y)
    plot_iterations(histories)
    run_grid_convergence_study(schemes, eps)


if __name__ == "__main__":
    main()
