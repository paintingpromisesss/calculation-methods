import math
import os
import matplotlib.pyplot as plt

EPS = 1e-12

# Физические параметры задачи для Варианта 1:
# A_COEF = 1.0 — скорость распространения волны (фазовая скорость a = 1.0).
# L = math.pi — расчетная область x in [0, pi].
# T = 1.0 — расчетный интервал времени; за это время гармоническая волна успевает
# сместиться на фазовый угол a*T = 1 радиан, наглядно демонстрируя волновой перенос профиля.
A_COEF = 1.0
L = math.pi
T = 1.0


def get_plot_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def exact_solution(x, t, a=A_COEF):
    # Аналитическое решение одномерного волнового уравнения для Варианта 1:
    # U(x, t) = sin(x - a * t)
    return math.sin(x - a * t)


def initial_grid(n, k_total, a=A_COEF, t_max=T, init_order=2):
    if n < 2 or k_total < 1 or a <= 0.0 or t_max <= 0.0:
        raise ValueError("Параметры сетки должны быть положительными")

    h = L / n
    tau = t_max / k_total
    sigma = (a * tau / h) ** 2  # Квадрат сеточного числа Куранта (параметр схемы)

    x_values = [j * h for j in range(n + 1)]
    t_values = [k * tau for k in range(k_total + 1)]

    # Начальное распределение при t = 0 (1-е начальное условие): u(x, 0) = sin(x)
    u_0 = [math.sin(x) for x in x_values]

    # Граничные условия Дирихле при t = 0: u(0, 0) = -sin(0) = 0, u(pi, 0) = sin(0) = 0
    u_0[0] = 0.0
    u_0[n] = 0.0

    # Второй временной слой t = tau формируется из 2-го начального условия u_t(x, 0) = -a*cos(x)
    u_1 = [0.0] * (n + 1)
    if init_order == 1:
        # Аппроксимация 1-го порядка: u_j^1 = u_j^0 + tau * u_t(x_j, 0) + O(tau)
        for j in range(1, n):
            u_1[j] = u_0[j] - tau * a * math.cos(x_values[j])
    else:
        # Аппроксимация 2-го порядка через ряд Тейлора:
        # u(x, tau) = u(x, 0) + tau * u_t(x, 0) + (tau^2 / 2) * u_tt(x, 0) + O(tau^3),
        # где из уравнения u_tt = a^2 * u_xx: u_tt(x, 0) = a^2 * (sin(x))'' = -a^2 * sin(x) = -a^2 * u_0
        for j in range(1, n):
            u_1[j] = (1.0 - 0.5 * (a * tau) ** 2) * u_0[j] - tau * a * math.cos(x_values[j])

    # Граничные условия Дирихле (1-го рода) при t = tau:
    # u(0, tau) = -sin(a * tau), u(pi, tau) = sin(a * tau)
    u_1[0] = -math.sin(a * tau)
    u_1[n] = math.sin(a * tau)

    return x_values, t_values, u_0, u_1, h, tau, sigma


def tridiagonal_solve(lower, middle, upper, right):
    """
    Метод прогонки (алгоритм Томаса) для решения трехдиагональной СЛАУ A*u = d.
    Условие применимости (диагональное преобладание): |middle_i| >= |lower_i| + |upper_i|.
    В наших неявных схемах оно выполняется строго, деление на ноль исключено.
    """
    m = len(middle)
    p = [0.0] * m  # Прогоночные коэффициенты alpha
    q = [0.0] * m  # Прогоночные коэффициенты beta
    res = [0.0] * m

    if abs(middle[0]) < EPS:
        raise ValueError("Деление на ноль на первом шаге прогонки")

    # Прямой ход прогонки: расчет прогоночных коэффициентов
    p[0] = -upper[0] / middle[0]
    q[0] = right[0] / middle[0]

    for i in range(1, m):
        denom = middle[i] + lower[i] * p[i - 1]
        if abs(denom) < EPS:
            raise ValueError("Деление на ноль в прямом ходе прогонки")
        p[i] = -upper[i] / denom if i < m - 1 else 0.0
        q[i] = (right[i] - lower[i] * q[i - 1]) / denom

    # Обратный ход прогонки: восстановление решения u_i = p_i * u_{i+1} + q_i
    res[m - 1] = q[m - 1]
    for i in range(m - 2, -1, -1):
        res[i] = p[i] * res[i + 1] + q[i]

    return res


def compute_errors(layers, x_values, t_values, a=A_COEF):
    n = len(x_values) - 1
    k_total = len(t_values) - 1

    # 1. Максимальная абсолютная погрешность на финальном временном слое t = T
    u_last = layers[-1]
    t_last = t_values[-1]
    max_err = max(abs(u_last[j] - exact_solution(x_values[j], t_last, a)) for j in range(n + 1))

    # 2. Интегральная среднеквадратичная погрешность (RMSE) по всем узлам расчетной сетки
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
    t_fixed = T
    x_fixed_idx = round(len(x_values) / 4)
    x_fixed = x_values[x_fixed_idx]

    # График 1: Профиль u(x, T) и временная динамика в фиксированной точке x = pi/4
    fig1, axes1 = plt.subplots(1, 2, num="Сравнение схем", figsize=(12, 4))
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
    axes1[0].legend()

    exact_t = [exact_solution(x_fixed, t, a) for t in t_values]
    axes1[1].plot(t_values, exact_t, "k--", label="Точное")
    axes1[1].set_title(f"Решение при x = {x_fixed:.3f}")
    axes1[1].set_xlabel("t")
    axes1[1].set_ylabel(f"u(x = {x_fixed:.3f}, t)")
    axes1[1].grid(True)
    axes1[1].legend()

    plt.tight_layout()
    plt.savefig(get_plot_path("plot_compare.png"), dpi=300, bbox_inches="tight")
    plt.close()

    # График 2: Профили бегущей синусоидальной волны для различных моментов времени
    times = [0.00, 0.25, 0.50, 1.00]
    fig2, axes2 = plt.subplots(2, 2, num="Профили волны (2х2)", figsize=(11, 8), sharey=True)
    dt = t_values[1] - t_values[0]

    plots_config = [
        ("Точное решение", None, axes2[0, 0]),
        ("Явная схема (крест)", solutions["Явная (крест)"], axes2[0, 1]),
        ("Неявная схема", solutions["Неявная"], axes2[1, 0]),
        ("Явная (1-й пор. НУ)", solutions["Явная (1-й пор. НУ)"], axes2[1, 1]),
    ]

    for title, layers, ax in plots_config:
        for t_val in times:
            k_idx = round(t_val / dt)
            exact_vals = [exact_solution(x, t_val, a) for x in x_values]

            if layers is None:
                ax.plot(x_values, exact_vals, label=f"t = {t_val:.2f}")
            else:
                line, = ax.plot(x_values, layers[k_idx], label=f"t = {t_val:.2f}")
                ax.plot(x_values, exact_vals, linestyle="--", color=line.get_color(), alpha=0.6)

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_ylim(-1.2, 1.2)
        ax.grid(True)
        ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(get_plot_path("plot_grid_2x2.png"), dpi=300, bbox_inches="tight")
    plt.close()


def run_grid_convergence_study(schemes):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), num="Исследование сходимости")

    # 1. Измельчение пространственного шага h при фиксированном числе Куранта nu = 0.5
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

    axes[0].set_title("Измельчение пространственной сетки")
    axes[0].set_xlabel("h = pi/N (tau = 0.5 h/a)")
    axes[0].set_ylabel("max |ошибка| при T")
    axes[0].grid(True, which="both")
    axes[0].legend()

    # 2. Измельчение шага по времени tau при фиксированном h (N = 40)
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
