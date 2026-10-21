import math
import os
import matplotlib.pyplot as plt

EPS = 1e-12

A_COEF = 1.0
L = math.pi
T = 1.0


def get_plot_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def exact_solution(x, t, a=A_COEF):
    # U(x, t) = sin(x - a * t)
    return math.sin(x - a * t)


def initial_grid(n, k_total, a=A_COEF, t_max=T, init_order=2):
    if n < 2 or k_total < 1 or a <= 0.0 or t_max <= 0.0:
        raise ValueError("Параметры сетки должны быть положительными")

    h = L / n
    tau = t_max / k_total
    sigma = (a * tau / h) ** 2

    x_values = [j * h for j in range(n + 1)]
    t_values = [k * tau for k in range(k_total + 1)]

    u_0 = [math.sin(x) for x in x_values]

    u_0[0] = 0.0
    u_0[n] = 0.0

    # Первый шаг по времени: u_t(x, 0) = -a*cos(x)
    u_1 = [0.0] * (n + 1)
    if init_order == 1:
        # Первый порядок
        for j in range(1, n):
            u_1[j] = u_0[j] - tau * a * math.cos(x_values[j])
    else:
        # Второй порядок: ряд Тейлора с u_tt(x, 0) = -a^2*sin(x)
        for j in range(1, n):
            u_1[j] = (1.0 - 0.5 * (a * tau) ** 2) * u_0[j] - tau * a * math.cos(x_values[j])

    # Граничные условия Дирихле
    u_1[0] = -math.sin(a * tau)
    u_1[n] = math.sin(a * tau)

    return x_values, t_values, u_0, u_1, h, tau, sigma


def tridiagonal_solve(lower, middle, upper, right):
    m = len(middle)
    p = [0.0] * m
    q = [0.0] * m
    res = [0.0] * m

    if abs(middle[0]) < EPS:
        raise ValueError("Деление на ноль на первом шаге прогонки")

    # Прямой ход
    p[0] = -upper[0] / middle[0]
    q[0] = right[0] / middle[0]

    for i in range(1, m):
        denom = middle[i] + lower[i] * p[i - 1]
        if abs(denom) < EPS:
            raise ValueError("Деление на ноль в прямом ходе прогонки")
        p[i] = -upper[i] / denom if i < m - 1 else 0.0
        q[i] = (right[i] - lower[i] * q[i - 1]) / denom

    # Обратный ход
    res[m - 1] = q[m - 1]
    for i in range(m - 2, -1, -1):
        res[i] = p[i] * res[i + 1] + q[i]

    return res


def compute_errors(layers, x_values, t_values, a=A_COEF):
    n = len(x_values) - 1
    k_total = len(t_values) - 1

    # Максимальная ошибка на последнем слое
    u_last = layers[-1]
    t_last = t_values[-1]
    max_err = max(abs(u_last[j] - exact_solution(x_values[j], t_last, a)) for j in range(n + 1))

    # RMSE по всем переданным узлам и слоям
    sum_sq = 0.0
    total = (k_total + 1) * (n + 1)
    for k in range(k_total + 1):
        t_curr = t_values[k]
        for j in range(n + 1):
            diff = layers[k][j] - exact_solution(x_values[j], t_curr, a)
            sum_sq += diff * diff

    rmse = math.sqrt(sum_sq / total)
    return max_err, rmse


def build_plots(solutions, x_values, t_values, a=A_COEF):
    t_fixed = t_values[-1]
    x_fixed_idx = round(len(x_values) / 4)
    x_fixed = x_values[x_fixed_idx]

    # Профиль при t = T и динамика в фиксированной точке
    _, axes1 = plt.subplots(1, 2, num="Сравнение схем", figsize=(12, 4))
    for name, layers in solutions.items():
        axes1[0].plot(x_values, layers[-1], label=name)
        u_t = [layers[k][x_fixed_idx] for k in range(len(t_values))]
        axes1[1].plot(t_values, u_t, label=name)

    exact_x = [exact_solution(x, t_fixed, a) for x in x_values]
    axes1[0].plot(x_values, exact_x, "k--", label="Точное")
    axes1[0].set_title(f"Решение при T = {t_fixed}")
    axes1[0].set_xlabel("x")
    axes1[0].set_ylabel("u(x, T)")
    axes1[0].grid(True)
    exact_first_legend(axes1[0])

    exact_t = [exact_solution(x_fixed, t, a) for t in t_values]
    axes1[1].plot(t_values, exact_t, "k--", label="Точное")
    axes1[1].set_title(f"Решение при x = {x_fixed:.3f}")
    axes1[1].set_xlabel("t")
    axes1[1].set_ylabel(f"u(x = {x_fixed:.3f}, t)")
    axes1[1].grid(True)
    exact_first_legend(axes1[1])

    plt.tight_layout()
    plt.savefig(get_plot_path("plot_compare.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # Профили волны в разные моменты времени
    indices = [round(fraction * (len(t_values) - 1)) for fraction in (0.0, 0.25, 0.5, 0.75, 1.0)]
    fig2, axes2 = plt.subplots(2, 2, num="Профили волны (2х2)", figsize=(11, 8), sharey=True)

    for (title, layers), ax in zip(solutions.items(), axes2.flat):
        for k_idx in indices:
            t_val = t_values[k_idx]
            exact_vals = [exact_solution(x, t_val, a) for x in x_values]

            ax.plot(x_values, layers[k_idx], linewidth=2, label=f"t = {t_val:.2f}")
            ax.plot(x_values, exact_vals, "k--", linewidth=1.2,
                    label="Точное решение" if k_idx == indices[0] else "_nolegend_")

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True)
        exact_first_legend(ax, loc="upper right")

    fig2.suptitle("Цветные линии — численное решение, чёрный пунктир — точное")
    plt.tight_layout()
    plt.savefig(get_plot_path("plot_grid_2x2.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_grid_convergence_study(schemes):
    _, axes = plt.subplots(1, 2, figsize=(12, 4.5), num="Исследование сходимости")

    # Совместное измельчение h и tau, число Куранта примерно 0.5
    n_list = [10, 20, 40, 80]
    nu_fixed = 0.5
    h_vals = [L / n for n in n_list]
    err_h = {name: [] for name, _ in schemes}

    for n in n_list:
        h = L / n
        tau = nu_fixed * h / A_COEF
        k = round(T / tau)
        for name, solver in schemes:
            x, t, layers, _ = solver(n=n, k_total=k)
            max_err, _ = compute_errors(layers, x, t)
            err_h[name].append(max_err)

    for name, _ in schemes:
        axes[0].loglog(h_vals, err_h[name], marker="o", label=name)

    axes[0].set_title("Совместное измельчение h и tau")
    axes[0].set_xlabel("h = pi/N (tau ≈ 0.5 h/a)")
    axes[0].set_ylabel("max |ошибка| при T")
    axes[0].grid(True, which="both")
    axes[0].legend()

    # Измельчение tau при фиксированном h
    n_fixed = 40
    k_list = [50, 100, 200, 400]
    tau_vals = [T / k for k in k_list]
    err_tau = {name: [] for name, _ in schemes}

    for k in k_list:
        for name, solver in schemes:
            x, t, layers, _ = solver(n=n_fixed, k_total=k)
            max_err, _ = compute_errors(layers, x, t)
            err_tau[name].append(max_err)

    for name, _ in schemes:
        axes[1].loglog(tau_vals, err_tau[name], marker="o", label=name)

    axes[1].set_title("Измельчение шага по времени")
    axes[1].set_xlabel(f"tau = T/K (h = pi/{n_fixed})")
    axes[1].set_ylabel("max |ошибка| при T")
    axes[1].grid(True, which="both")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(get_plot_path("plot_convergence.png"), dpi=300)
    plt.close()

def exact_first_legend(ax, **kwargs):
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(
        range(len(labels)),
        key=lambda i: 0 if labels[i].startswith("Точное") else 1
    )
    ax.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        **kwargs
    )

def plot_error_over_time(solutions, x_values, t_values, a=A_COEF):
    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        num="Погрешность в зависимости от времени"
    )

    for name, layers in solutions.items():
        max_errors = []
        rmse_errors = []

        for k, t in enumerate(t_values):
            errors = [
                layers[k][j] - exact_solution(x, t, a)
                for j, x in enumerate(x_values)
            ]

            max_errors.append(max(abs(e) for e in errors))
            rmse_errors.append(
                math.sqrt(sum(e**2 for e in errors) / len(errors))
            )

        axes[0].plot(t_values, max_errors, linewidth=2, label=name)
        axes[1].plot(t_values, rmse_errors, linewidth=2, label=name)

    axes[0].set_title("Максимальная абсолютная погрешность")
    axes[0].set_ylabel("max |u(x,t) - U(x,t)|")

    axes[1].set_title("Среднеквадратичная погрешность")
    axes[1].set_ylabel("RMSE")

    for ax in axes:
        ax.set_xlabel("Время t")
        ax.set_xlim(t_values[0], t_values[-1])
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.4)
        ax.legend(loc="upper left")

    fig.suptitle(
        "Зависимость погрешности численного решения от времени",
        fontsize=14
    )

    plt.tight_layout()
    plt.savefig(
        get_plot_path("plot_error_over_time.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()
