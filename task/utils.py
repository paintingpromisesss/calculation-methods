import math
import os
import matplotlib.pyplot as plt

L_X = 1.0
L_Y = 1.0


def get_plot_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def exact_solution(x, y):
    return x + y


def initial_grid(n_x, n_y):
    if n_x < 2 or n_y < 2:
        raise ValueError("По каждой координате необходимо не менее двух интервалов")
    h_x = L_X / n_x
    h_y = L_Y / n_y
    x_values = [i * h_x for i in range(n_x + 1)]
    y_values = [j * h_y for j in range(n_y + 1)]
    u = [[0.0] * (n_x + 1) for _ in range(n_y + 1)]

    for j, y in enumerate(y_values):
        u[j][0] = y
        u[j][-1] = 1.0 + y
    for i, x in enumerate(x_values):
        u[0][i] = x
        u[-1][i] = 1.0 + x

    return x_values, y_values, u, h_x, h_y


def compute_errors(u, x_values, y_values):
    max_err = 0.0
    sum_sq = 0.0
    for j, y in enumerate(y_values):
        for i, x in enumerate(x_values):
            diff = u[j][i] - exact_solution(x, y)
            max_err = max(max_err, abs(diff))
            sum_sq += diff**2
    rmse = math.sqrt(sum_sq / (len(x_values) * len(y_values)))
    return max_err, rmse


def build_plots(solutions, x_values, y_values):
    i_fixed = (len(x_values) - 1) // 2
    j_fixed = (len(y_values) - 1) // 2
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), num="Сравнение методов")
    axes[0].plot(x_values, [exact_solution(x, y_values[j_fixed]) for x in x_values],
                 "k--", label="Точное решение", zorder=5)
    axes[1].plot(y_values, [exact_solution(x_values[i_fixed], y) for y in y_values],
                 "k--", label="Точное решение", zorder=5)
    for name, u in solutions.items():
        axes[0].plot(x_values, u[j_fixed], label=name)
        axes[1].plot(y_values, [row[i_fixed] for row in u], label=name)
    axes[0].set_title(f"Решение при y = {y_values[j_fixed]:.2f}")
    axes[0].set_xlabel("x")
    axes[1].set_title(f"Решение при x = {x_values[i_fixed]:.2f}")
    axes[1].set_xlabel("y")
    for ax in axes:
        ax.set_ylabel("u(x, y)")
        ax.grid(True)
        ax.legend()
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_compare.png"), dpi=300)
    plt.close(fig)

    exact = [[exact_solution(x, y) for x in x_values] for y in y_values]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), num="Решения (2х2)")
    for (name, u), ax in zip([("Точное решение", exact)] + list(solutions.items()), axes.flat):
        image = ax.imshow(u, origin="lower", extent=(0, L_X, 0, L_Y),
                          vmin=0, vmax=2, aspect="equal")
        ax.set_title(name)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        fig.colorbar(image, ax=ax, label="u(x, y)")
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_grid_2x2.png"), dpi=300)
    plt.close(fig)


def plot_iterations(histories):
    fig, ax = plt.subplots(figsize=(8, 5), num="Сходимость итераций")
    for name, history in histories.items():
        ax.semilogy(range(1, len(history) + 1), history, label=name)
    ax.set_xlabel("Номер итерации")
    ax.set_ylabel("max |u^(k+1) − u^k|")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_iterations.png"), dpi=300)
    plt.close(fig)


def run_grid_convergence_study(schemes, eps=1e-8):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), num="Исследование сетки")
    n_list = [10, 20, 40]
    for column, coordinate in enumerate(("x", "y")):
        print(f"\nИзмельчение h_{coordinate}, по другой координате N=20, eps={eps:g}")
        print(f"{'N':<6}{'Метод':<30}{'Итерации':<12}{'max|ошибка|':<18}{'RMSE'}")
        for name, solver in schemes:
            errors = []
            iterations = []
            for n in n_list:
                n_x, n_y = (n, 20) if coordinate == "x" else (20, n)
                x, y, u, history = solver(n_x, n_y, eps)
                max_err, rmse = compute_errors(u, x, y)
                errors.append(max_err)
                iterations.append(len(history))
                print(f"{n:<6}{name:<30}{len(history):<12}{max_err:<18.6e}{rmse:.6e}")
            h_values = [1.0 / n for n in n_list]
            axes[0, column].loglog(h_values, errors, marker="o", label=name)
            axes[1, column].semilogx(h_values, iterations, marker="o", label=name)
        axes[0, column].set_title(f"Измельчение h_{coordinate} при фиксированном другом шаге")
        axes[0, column].set_ylabel("max |u − U|")
        axes[1, column].set_ylabel("Число итераций")
        for ax in axes[:, column]:
            ax.set_xlabel(f"h_{coordinate}")
            ax.grid(True, which="both")
            ax.legend()
    fig.suptitle("U = x + y: погрешность остановки итераций, eps = 1e-8")
    fig.tight_layout()
    fig.savefig(get_plot_path("plot_convergence.png"), dpi=300)
    plt.close(fig)
