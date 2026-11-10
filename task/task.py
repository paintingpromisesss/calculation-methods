from utils import (
    A_COEF,
    L_X,
    L_Y,
    T,
    initial_grid,
    boundary_grid,
    tridiagonal_solve,
    compute_errors,
    build_plots,
    plot_error_over_time,
    run_grid_convergence_study,
)


def solve_splitting(n_x=24, n_y=24, k_total=100, a=A_COEF, t_max=T,
                    mu_1=1, mu_2=1, variable_directions=True):
    x_values, y_values, t_values, u, h_x, h_y, tau = initial_grid(
        n_x, n_y, k_total, a, t_max, mu_1, mu_2
    )

    # МПН: tau/2 в (5.78)-(5.79); МДШ: полный tau в (5.80)-(5.81).
    step = tau / 2.0 if variable_directions else tau
    sigma_x = a * step / h_x**2
    sigma_y = a * step / h_y**2
    lower_x = [-sigma_x] * (n_x - 1)
    middle_x = [1.0 + 2.0 * sigma_x] * (n_x - 1)
    upper_x = [-sigma_x] * (n_x - 1)
    lower_y = [-sigma_y] * (n_y - 1)
    middle_y = [1.0 + 2.0 * sigma_y] * (n_y - 1)
    upper_y = [-sigma_y] * (n_y - 1)
    layers = [u]

    for k in range(k_total):
        t_half = t_values[k] + tau / 2.0
        u_half = boundary_grid(x_values, y_values, t_half, a, mu_1, mu_2)
        u_next = boundary_grid(x_values, y_values, t_values[k + 1], a, mu_1, mu_2)

        # Прогонки вдоль x: в МПН производная по y берется явно.
        for j in range(1, n_y):
            right = [u[j][i] for i in range(1, n_x)]
            if variable_directions:
                for i in range(1, n_x):
                    right[i - 1] += sigma_y * (u[j - 1][i] - 2.0*u[j][i] + u[j + 1][i])
            right[0] += sigma_x * u_half[j][0]
            right[-1] += sigma_x * u_half[j][-1]
            u_half[j][1:n_x] = tridiagonal_solve(lower_x, middle_x, upper_x, right)

        # Прогонки вдоль y: в МПН производная по x берется с полуслоя.
        for i in range(1, n_x):
            right = [u_half[j][i] for j in range(1, n_y)]
            if variable_directions:
                for j in range(1, n_y):
                    right[j - 1] += sigma_x * (u_half[j][i - 1] - 2.0*u_half[j][i] + u_half[j][i + 1])
            right[0] += sigma_y * u_next[0][i]
            right[-1] += sigma_y * u_next[-1][i]
            internal = tridiagonal_solve(lower_y, middle_y, upper_y, right)
            for j in range(1, n_y):
                u_next[j][i] = internal[j - 1]

        u = u_next
        layers.append(u)

    return x_values, y_values, t_values, layers


def solve_variable_directions(n_x=24, n_y=24, k_total=100, a=A_COEF,
                              t_max=T, mu_1=1, mu_2=1):
    return solve_splitting(n_x, n_y, k_total, a, t_max, mu_1, mu_2,
                           variable_directions=True)


def solve_fractional_steps(n_x=24, n_y=24, k_total=100, a=A_COEF,
                           t_max=T, mu_1=1, mu_2=1):
    return solve_splitting(n_x, n_y, k_total, a, t_max, mu_1, mu_2,
                           variable_directions=False)


def main():
    n_x = 24
    n_y = 24
    k_total = 100
    schemes = [
        ("Переменные направления", solve_variable_directions),
        ("Дробные шаги", solve_fractional_steps),
    ]

    all_solutions = {}
    for mu_1, mu_2 in ((1, 1), (2, 1), (1, 2)):
        print(f"\nmu_1={mu_1}, mu_2={mu_2}, a={A_COEF:g}")
        print(f"Nx={n_x}, Ny={n_y}, K={k_total}, hx={L_X/n_x:.5f}, hy={L_Y/n_y:.5f}, tau={T/k_total:.5f}")
        print(f"{'Метод':<26}{'t':<10}{'max|ошибка|':<18}{'RMSE'}")
        solutions = {}
        for name, solver in schemes:
            x, y, t, layers = solver(n_x, n_y, k_total, mu_1=mu_1, mu_2=mu_2)
            solutions[name] = layers
            for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                k = round(fraction * k_total)
                max_err, rmse = compute_errors([layers[k]], x, y, [t[k]], mu_1=mu_1, mu_2=mu_2)
                print(f"{name:<26}{t[k]:<10.2f}{max_err:<18.6e}{rmse:.6e}")

        all_solutions[mu_1, mu_2] = solutions

    build_plots(all_solutions, x, y, t)
    plot_error_over_time(all_solutions, x, y, t)
    run_grid_convergence_study(schemes, all_solutions)


if __name__ == "__main__":
    main()
