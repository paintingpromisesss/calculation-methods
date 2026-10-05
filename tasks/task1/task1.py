import math

from utils import (
    A,
    B,
    MU,
    L_X,
    L_Y,
    T,
    get_grid,
    get_max_error,
    print_research_table,
    print_solution_table,
    set_x_boundary,
    set_y_boundary,
    source,
    tridiagonal_solve,
)


def sweep_x(u, u_result, h_x, h_y, dt, t_src, coeff, f_half, src_scale, t_bnd):
    # Прогонки вдоль строк (по x) с шагом dt.
    # coeff = a * dt / h_x^2 - коэффициент прогонки.
    # f_half = True, если в правую часть входит явный оператор по y.
    # t_bnd - момент для граничного условия u_x(pi, y, t) = g(y, t).
    x_values = get_grid(L_X, h_x)
    y_values = get_grid(L_Y, h_y)

    i_max = len(x_values) - 1
    j_max = len(y_values) - 1

    n = i_max - 1

    lower = [-coeff for _ in range(n)]
    # Условие u_x(pi, y, t) = g аппроксимируем двухточечно:
    # (u_I - u_{I-1}) / h_x = g  =>  u_I = u_{I-1} + h_x g.
    # Подстановка в уравнение при i = I-1:
    # -c u_{I-2} + (1 + c) u_{I-1} = rhs + c h_x g.
    # Условие u(0, y, t) = 0 входит в первое уравнение без изменения.
    middle = [1.0 + 2.0 * coeff for _ in range(n)]
    middle[n - 1] = 1.0 + coeff
    upper = [-coeff for _ in range(n)]

    for j in range(1, j_max):
        y = y_values[j]

        right = []

        for i in range(1, i_max):
            x = x_values[i]

            # u^k + dt * (явная часть по y для МПН + источник)
            value = u[i][j] + dt * src_scale * source(x, y, t_src)

            if f_half:
                value += dt * B * (u[i][j + 1] - 2 * u[i][j] + u[i][j - 1]) / h_y**2

            right.append(value)

        # Граничное условие второго рода при x = pi
        right[n - 1] += coeff * h_x * (-math.sin(MU * t_bnd)) * math.sin(y)

        row = tridiagonal_solve(lower, middle, upper, right)

        for i in range(1, i_max):
            u_result[i][j] = row[i - 1]


def sweep_y(u, u_result, h_x, h_y, dt, t_src, coeff, f_half, src_scale, t_bnd):
    # Прогонки вдоль столбцов (по y) с шагом dt
    x_values = get_grid(L_X, h_x)
    y_values = get_grid(L_Y, h_y)

    i_max = len(x_values) - 1
    j_max = len(y_values) - 1

    n = j_max - 1

    lower = [-coeff for _ in range(n)]
    # Аналогично по y: u_y(x, pi, t) = g => u_J = u_{J-1} + h_y g
    middle = [1.0 + 2.0 * coeff for _ in range(n)]
    middle[n - 1] = 1.0 + coeff
    upper = [-coeff for _ in range(n)]

    for i in range(1, i_max):
        x = x_values[i]

        right = []

        for j in range(1, j_max):
            y = y_values[j]

            # u^{пред. слой} + dt * (явная часть по x для МПН + источник)
            value = u[i][j] + dt * src_scale * source(x, y, t_src)

            if f_half:
                value += dt * A * (u[i + 1][j] - 2 * u[i][j] + u[i - 1][j]) / h_x**2

            right.append(value)

        # Граничное условие второго рода при y = pi
        right[n - 1] += coeff * h_y * (-math.sin(MU * t_bnd)) * math.sin(x)

        column = tridiagonal_solve(lower, middle, upper, right)

        for j in range(1, j_max):
            u_result[i][j] = column[j - 1]


def solve_splitting(h_x, h_y, tau, method, t_end=T):
    # Решение задачи схемой расщепления до момента t_end.
    # method = "mpn" - метод переменных направлений (5.78)-(5.79),
    #          "mdsh" - метод дробных шагов (5.80)-(5.81).
    if method not in ("mpn", "mdsh"):
        raise ValueError("Неизвестный метод: ожидается 'mpn' или 'mdsh'")

    x_values = get_grid(L_X, h_x)
    y_values = get_grid(L_Y, h_y)

    i_max = len(x_values) - 1
    j_max = len(y_values) - 1

    n_steps = int(round(t_end / tau))

    if abs(t_end / tau - n_steps) > 1e-9:
        raise ValueError("Момент времени не кратен шагу tau")

    # Начальное условие u(x, y, 0) = 0
    u = [
        [0.0 for _ in range(j_max + 1)]
        for _ in range(i_max + 1)
    ]

    # Шаги по времени: МПН - два полушага tau/2, МДШ - два шага tau
    dt = tau / 2 if method == "mpn" else tau

    # Коэффициенты прогонки (не зависят от номера слоя)
    coeff_x = A * dt / h_x**2
    coeff_y = B * dt / h_y**2

    for k in range(n_steps):
        t_k = k * tau
        t_half = t_k + tau / 2
        t_next = t_k + tau

        u_half = [
            [0.0 for _ in range(j_max + 1)]
            for _ in range(i_max + 1)
        ]

        # ---- Полушаг 1: неявно по x, прогонки вдоль строк ----
        # Источник: МПН - на полу-слое t_{k+1/2}, МДШ - f^k / 2
        if method == "mpn":
            t_src = t_half
            f_half = True
            # Граничное условие u_x(pi, y, t) - на полу-слое t_{k+1/2}
            t_bnd_x = t_half
        else:
            t_src = t_k
            f_half = False
            # Для МДШ граничное условие по x берем на слое t_{k+1}
            t_bnd_x = t_next

        # Источник: МПН - f^{k+1/2} целиком, МДШ - f^k / 2
        src_scale = 1.0 if method == "mpn" else 0.5

        sweep_x(
            u, u_half, h_x, h_y, dt, t_src, coeff_x, f_half, src_scale, t_bnd_x
        )

        # Граница x = pi на полуслое (согласована с условием в прогонке)
        set_x_boundary(u_half, x_values, y_values, t_bnd_x)

        # Границы по y на полуслое: МПН - уже заданы, МДШ - слой k
        if method == "mdsh":
            for i in range(i_max + 1):
                u_half[i][0] = u[i][0]
                u_half[i][j_max] = u[i][j_max]

        u_next = [
            [0.0 for _ in range(j_max + 1)]
            for _ in range(i_max + 1)
        ]

        # ---- Полушаг 2: неявно по y, прогонки вдоль столбцов ----
        # Источник: МПН - f^{k+1/2}, МДШ - f^{k+1} / 2
        # Граничное условие u_y(x, pi, t) - на слое t_{k+1}
        if method == "mpn":
            t_src = t_half
            f_half = True
        else:
            t_src = t_next
            f_half = False

        sweep_y(
            u_half, u_next, h_x, h_y, dt, t_src, coeff_y, f_half, src_scale,
            t_next
        )

        # Границы по y на слое k+1
        set_y_boundary(u_next, x_values, y_values, t_next)

        # Границы по x на слое k+1
        set_x_boundary(u_next, x_values, y_values, t_next)

        u = u_next

    return u, x_values, y_values


def main():
    h_x = L_X / 20
    h_y = L_Y / 20
    tau = 0.01

    print("Лабораторная работа 8. Вариант 10")
    print("u_t = a u_xx + b u_yy + sin x sin y (cos mu t + (a+b) sin mu t)")
    print(f"a = {A:g}, b = {B:g}, mu = {MU:g}")
    print(f"Область: x, y из [0, pi], T = {T:g}")
    print("u(0,y,t) = 0, u_x(pi,y,t) = -sin y sin(mu t)")
    print("u(x,0,t) = 0, u_y(x,pi,t) = -sin x sin(mu t)")
    print("u(x,y,0) = 0")
    print("Аналитическое решение: U = sin x sin y sin(mu t)")
    print()

    for method, name in (("mpn", "МПН"), ("mdsh", "МДШ")):
        print(f"Схема {name}, h_x = h_y = {h_x:.4f}, tau = {tau:g}")
        print()

        for t in (0.1, 0.5, 1.0):
            u, x_values, y_values = solve_splitting(
                h_x, h_y, tau, method, t_end=t
            )

            print_solution_table(
                f"Схема {name}:",
                u,
                x_values,
                y_values,
                t,
            )

    # ---- Исследование погрешности от tau (h фиксированы) ----
    # h берем мелким, чтобы пространственная погрешность не забивала
    # временную сходимость МПН O(tau^2)
    h_fixed = L_X / 160

    print("Исследование зависимости погрешности от tau")
    print(f"h_x = h_y = {h_fixed:.4f} фиксированы, момент T = {T:g}")
    print()

    for method, name in (("mpn", "МПН"), ("mdsh", "МДШ")):
        rows = []

        for n_tau in (10, 20, 40):
            tau_cur = T / n_tau

            u, xs, ys = solve_splitting(h_fixed, h_fixed, tau_cur, method)
            rows.append((tau_cur, get_max_error(u, xs, ys, T)))

        print_research_table(f"Схема {name}:", rows)

    # ---- Исследование погрешности от h (tau фиксировано) ----
    tau_fixed = 0.0005

    print("Исследование зависимости погрешности от h_x = h_y")
    print(f"tau = {tau_fixed:g} фиксировано, момент T = {T:g}")
    print()

    for method, name in (("mpn", "МПН"), ("mdsh", "МДШ")):
        rows = []

        for n_h in (5, 10, 20, 40):
            h_cur = L_X / n_h

            u, xs, ys = solve_splitting(h_cur, h_cur, tau_fixed, method)
            rows.append((h_cur, get_max_error(u, xs, ys, T)))

        print_research_table(f"Схема {name}:", rows)


if __name__ == "__main__":
    main()
