import math
import os
import matplotlib.pyplot as plt

EPS = 1e-12
A_COEF = 1.0
L_X = math.pi
L_Y = math.pi
T = 1.0


def get_plot_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def exact_solution(x, y, t, a=A_COEF, mu_1=1, mu_2=1):
    return math.cos(mu_1*x) * math.cos(mu_2*y) * math.exp(-a*(mu_1**2 + mu_2**2)*t)


def boundary_grid(x_values, y_values, t, a=A_COEF, mu_1=1, mu_2=1):
    u = [[0.0] * len(x_values) for _ in y_values]
    decay = math.exp(-a * (mu_1**2 + mu_2**2) * t)
    for j, y in enumerate(y_values):
        u[j][0] = math.cos(mu_2*y) * decay
        u[j][-1] = (-1)**mu_1 * math.cos(mu_2*y) * decay
    for i, x in enumerate(x_values):
        u[0][i] = math.cos(mu_1*x) * decay
        u[-1][i] = (-1)**mu_2 * math.cos(mu_1*x) * decay
    return u


def initial_grid(n_x, n_y, k_total, a=A_COEF, t_max=T, mu_1=1, mu_2=1):
    if any(not isinstance(n, int) or isinstance(n, bool) or n < minimum
           for n, minimum in ((n_x, 2), (n_y, 2), (k_total, 1), (mu_1, 1), (mu_2, 1))):
        raise ValueError("Nx, Ny >= 2, K >= 1; mu_1 и mu_2 — положительные целые")
    if not math.isfinite(a) or not math.isfinite(t_max) or a <= 0.0 or t_max <= 0.0:
        raise ValueError("Коэффициент a и конечное время должны быть положительными")

    h_x = L_X / n_x
    h_y = L_Y / n_y
    tau = t_max / k_total
    x_values = [i * h_x for i in range(n_x + 1)]
    y_values = [j * h_y for j in range(n_y + 1)]
    t_values = [k * tau for k in range(k_total + 1)]
    u = boundary_grid(x_values, y_values, 0.0, a, mu_1, mu_2)
    for j in range(1, n_y):
        for i in range(1, n_x):
            u[j][i] = math.cos(mu_1*x_values[i]) * math.cos(mu_2*y_values[j])
    return x_values, y_values, t_values, u, h_x, h_y, tau


def tridiagonal_solve(lower, middle, upper, right):
    m = len(middle)
    p = [0.0] * m
    q = [0.0] * m
    res = [0.0] * m
    if abs(middle[0]) < EPS:
        raise ValueError("Деление на ноль на первом шаге прогонки")
    p[0] = -upper[0] / middle[0]
    q[0] = right[0] / middle[0]

    for i in range(1, m):
        denom = middle[i] + lower[i] * p[i - 1]
        if abs(denom) < EPS:
            raise ValueError("Деление на ноль в прямом ходе прогонки")
        p[i] = -upper[i] / denom if i < m - 1 else 0.0
        q[i] = (right[i] - lower[i] * q[i - 1]) / denom

    res[-1] = q[-1]
    for i in range(m - 2, -1, -1):
        res[i] = p[i] * res[i + 1] + q[i]
    return res


def compute_errors(layers, x_values, y_values, t_values, a=A_COEF, mu_1=1, mu_2=1):
    max_err = 0.0
    sum_sq = 0.0
    for k, t in enumerate(t_values):
        for j, y in enumerate(y_values):
            for i, x in enumerate(x_values):
                diff = layers[k][j][i] - exact_solution(x, y, t, a, mu_1, mu_2)
                if k == len(t_values) - 1:
                    max_err = max(max_err, abs(diff))
                sum_sq += diff**2
    rmse = math.sqrt(sum_sq / (len(t_values) * len(x_values) * len(y_values)))
    return max_err, rmse


def build_plots(all_solutions, x_values, y_values, t_values, a=A_COEF):
    i_fixed = (len(x_values) - 1) // 4
    j_fixed = (len(y_values) - 1) // 4
    t_fixed = t_values[-1]
    fig, axes = plt.subplots(len(all_solutions), 2, figsize=(12, 10),
                             squeeze=False, num="Сравнение методов")
    for ((mu_1, mu_2), solutions), row in zip(all_solutions.items(), axes):
        label = f"μ₁={mu_1}, μ₂={mu_2}"
        for name, layers in solutions.items():
            row[0].plot(y_values, [line[i_fixed] for line in layers[-1]], label=name)
            row[1].plot(x_values, layers[-1][j_fixed], label=name)
        row[1].plot(x_values, [exact_solution(x, y_values[j_fixed], t_fixed, a, mu_1, mu_2)
                              for x in x_values], "k--", label="Точное решение")
        row[0].plot(y_values, [exact_solution(x_values[i_fixed], y, t_fixed, a, mu_1, mu_2)
                              for y in y_values], "k--", label="Точное решение")
        row[0].set_title(f"{label}, x={x_values[i_fixed]:.3f}")
        row[1].set_title(f"{label}, y={y_values[j_fixed]:.3f}")
        row[0].set_xlabel("y")
        row[1].set_xlabel("x")
        for ax in row:
            ax.set_ylabel("u(x, y, t)")
            ax.grid(True)
            ax.legend()
    fig.suptitle(f"Сравнение методов при t={t_fixed:g}")
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_compare.png"), dpi=300)
    plt.close(fig)

    indices = [round(fraction * (len(t_values) - 1)) for fraction in (0.0, 0.25, 0.5, 1.0)]
    fig, axes = plt.subplots(len(all_solutions), 4, figsize=(18, 10),
                             squeeze=False, num="Профили решения")
    for ((mu_1, mu_2), solutions), row in zip(all_solutions.items(), axes):
        for k, ax in zip(indices, row):
            for name, layers in solutions.items():
                ax.plot(x_values, layers[k][j_fixed], label=name)
            ax.plot(x_values, [exact_solution(x, y_values[j_fixed], t_values[k], a, mu_1, mu_2)
                               for x in x_values], "k--", label="Точное решение")
            ax.set_title(f"μ₁={mu_1}, μ₂={mu_2}, t={t_values[k]:.2f}")
            ax.set_xlabel("x")
            ax.set_ylabel("u(x, y, t)")
            ax.grid(True)
            ax.legend(fontsize=8)
    fig.suptitle(f"Профили решения при y={y_values[j_fixed]:.3f}")
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_profiles.png"), dpi=300)
    plt.close(fig)


def plot_error_over_time(all_solutions, x_values, y_values, t_values, a=A_COEF):
    fig, axes = plt.subplots(len(all_solutions), 2, figsize=(12, 10),
                             squeeze=False, num="Погрешность по времени")
    for ((mu_1, mu_2), solutions), row in zip(all_solutions.items(), axes):
        for name, layers in solutions.items():
            max_errors = []
            rmse_errors = []
            for k, t in enumerate(t_values):
                max_err, rmse = compute_errors([layers[k]], x_values, y_values, [t], a, mu_1, mu_2)
                max_errors.append(max_err)
                rmse_errors.append(rmse)
            row[0].plot(t_values, max_errors, label=name)
            row[1].plot(t_values, rmse_errors, label=name)
        row[0].set_ylabel("max |u − U|")
        row[1].set_ylabel("RMSE")
        for ax in row:
            ax.set_title(f"μ₁={mu_1}, μ₂={mu_2}")
            ax.set_xlabel("t")
            ax.set_ylim(bottom=0)
            ax.grid(True)
            ax.legend()
    fig.suptitle("Погрешность по времени")
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_error_over_time.png"), dpi=300)
    plt.close(fig)


def run_grid_convergence_study(schemes, cases):
    fig, axes = plt.subplots(len(cases), 3, figsize=(16, 10),
                             squeeze=False, num="Исследование сетки")
    studies = [
        ("hx", [(n, 40, 200) for n in (10, 20, 40)]),
        ("hy", [(40, n, 200) for n in (10, 20, 40)]),
        ("tau", [(40, 40, k) for k in (10, 20, 40, 80)]),
    ]
    for (mu_1, mu_2), row in zip(cases, axes):
        print(f"\nИсследование сетки: mu_1={mu_1}, mu_2={mu_2}")
        for ax, (parameter, grids) in zip(row, studies):
            print(f"\nИзменение {parameter}, остальные шаги фиксированы")
            print(f"{'Шаг':<12}{'Метод':<26}{'max|ошибка| при T':<24}{'RMSE при T'}")
            for name, solver in schemes:
                steps = []
                errors = []
                for n_x, n_y, k_total in grids:
                    x, y, t, layers = solver(n_x, n_y, k_total, mu_1=mu_1, mu_2=mu_2)
                    max_err, rmse = compute_errors([layers[-1]], x, y, [t[-1]], mu_1=mu_1, mu_2=mu_2)
                    step = {"hx": L_X/n_x, "hy": L_Y/n_y, "tau": T/k_total}[parameter]
                    steps.append(step)
                    errors.append(max_err)
                    print(f"{step:<12.5f}{name:<26}{max_err:<24.6e}{rmse:.6e}")
                ax.loglog(steps, errors, marker="o", label=name)
            ax.set_title(f"μ₁={mu_1}, μ₂={mu_2}: {parameter}")
            ax.set_xlabel(parameter)
            ax.set_ylabel("max |u − U| при T")
            ax.grid(True, which="both")
            ax.legend()
    fig.suptitle("Влияние шагов сетки")
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_convergence.png"), dpi=300)
    plt.close(fig)
