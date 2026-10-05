import math

from utils import (
    A_SQ,
    B,
    C,
    D,
    L,
    T,
    check_source_consistency,
    f,
    get_grid_x,
    get_max_error,
    get_time_steps,
    phi_left,
    phi_right,
    print_convergence_table,
    print_error_summary,
    print_layer_table,
    print_orders_table,
    psi1,
    psi2,
    tridiagonal_solve,
)


BOUNDARY_NAMES = (
    "двухточечная 1-й порядок",
    "трехточечная 2-й порядок",
    "двухточечная 2-й порядок (Тейлор)",
)


def first_layer(x_values, tau, order):
    # Первый слой из второго начального условия u_t(x, 0) = psi2.
    # 1-й порядок: u_j^1 = psi1 + tau psi2.
    # 2-й порядок (ряд Тейлора, 5.2.2):
    # u_j^1 = psi1 + tau psi2 + tau^2/2 * u_tt(x, 0),
    # u_tt(x, 0) = a^2 psi1'' + b psi1' + c psi1 - d psi2 + f(x, 0)
    #            = -sin x + cos x - sin x + 3 sin x - cos x = sin x.
    u = []

    for x in x_values:
        value = psi1(x) + tau * psi2(x)

        if order == 2:
            utt0 = (
                -psi1(x)
                + math.cos(x)
                + C * psi1(x)
                - D * psi2(x)
                + f(x, 0.0)
            )
            value += tau**2 / 2 * utt0

        u.append(value)

    return u


def set_boundary_first_order(u_new, t_new, h):
    # Двухточечная аппроксимация с 1-м порядком:
    # (u_1 - u_0) / h = phi_left(t),  (u_N - u_{N-1}) / h = phi_right(t).
    u_new[0] = u_new[1] - h * phi_left(t_new)
    u_new[-1] = u_new[-2] + h * phi_right(t_new)


def set_boundary_three_point(u_new, t_new, h):
    # Трехточечная аппроксимация со 2-м порядком:
    # (-3u_0 + 4u_1 - u_2) / (2h) = phi_left(t),
    # (u_{N-2} - 4u_{N-1} + 3u_N) / (2h) = phi_right(t).
    u_new[0] = (4 * u_new[1] - u_new[2] - 2 * h * phi_left(t_new)) / 3
    u_new[-1] = (4 * u_new[-2] - u_new[-3] + 2 * h * phi_right(t_new)) / 3


def taylor_boundary_equation(u_new, u_cur, u_prev, t_new, h, tau, left):
    # Двухточечная аппроксимация со 2-м порядком (метод Тейлора, 5.1.3).
    # Раскладываем u_1 (или u_{N-1}) по Тейлору в граничном узле:
    # u_1 = u_0 + h u_x(0) + h^2/2 u_xx(0) + O(h^3).
    # Вторую производную выражаем из ДУ:
    # u_xx = u_tt + d u_t - b u_x - c u - f,
    # u_tt и u_t аппроксимируем разностями по времени в самом граничном узле:
    # u_tt ~ (u_0^{k+1} - 2u_0^k + u_0^{k-1}) / tau^2,
    # u_t  ~ (u_0^{k+1} - u_0^k) / tau,  u_x(0) = phi(t).
    # Возвращаем коэффициент при u_0^{k+1} (aa) и свободный член (bb):
    # aa * u_0^{k+1} = u_1^{k+1} - h * phi - bb  (левая граница).
    if left:
        u_neighbor = u_new[1]
        u_center = u_cur[0]
        u_old = u_prev[0]
        phi = phi_left(t_new)
        f_value = f(0.0, t_new)
        h_sign = -h
    else:
        u_neighbor = u_new[-2]
        u_center = u_cur[-1]
        u_old = u_prev[-1]
        phi = phi_right(t_new)
        f_value = f(L, t_new)
        h_sign = h

    half = h * h / 2

    # Коэффициент при u_0^{k+1} из членов u_tt, d u_t, -c u.
    aa = 1 + half * (1 / tau**2 + D / tau - C)

    # Свободные члены: -half*(u_tt без u^{k+1}) - half*(d u_t без u^{k+1}) ...
    bb = half * (-2 * u_center + u_old) / tau**2
    bb += half * (-D * u_center / tau - B * phi - f_value)

    # Уравнение: aa * u_0^{k+1} = u_neighbor + h_sign * phi - bb.
    return aa, u_neighbor + h_sign * phi - bb


def set_boundary_taylor_explicit(u_new, u_cur, u_prev, t_new, h, tau):
    # Явная схема: граничное уравнение Тейлора содержит только u_0^{k+1}
    # из нового слоя, поэтому разрешается явно.
    aa, rhs = taylor_boundary_equation(
        u_new, u_cur, u_prev, t_new, h, tau, left=True
    )
    u_new[0] = rhs / aa

    aa, rhs = taylor_boundary_equation(
        u_new, u_cur, u_prev, t_new, h, tau, left=False
    )
    u_new[-1] = rhs / aa


def solve_explicit(n_x, tau, init_order, boundary_kind):
    # Явная схема крест (5.38) для уравнения
    # u_tt = a^2 u_xx + b u_x + c u - d u_t + f.
    # Производные по x - на слое k; член d u_t аппроксимируем
    # (u^k - u^{k-1}) / tau (односторонняя разность, стандартный прием
    # для уравнений с трением).
    h = L / n_x
    x_values = get_grid_x(n_x)

    n_t, tau = get_time_steps(tau)

    sigma = A_SQ * tau**2 / h**2

    if sigma >= 1:
        raise ValueError(
            f"Нарушено условие устойчивости sigma = tau^2/h^2 < 1 "
            f"(sigma = {sigma:.4f})"
        )

    u_prev = [psi1(x) for x in x_values]
    u_cur = first_layer(x_values, tau, init_order)

    layers = [u_prev, u_cur]

    for k in range(1, n_t):
        t_new = (k + 1) * tau
        t_cur = k * tau

        u_new = [0.0 for _ in range(n_x + 1)]

        for j in range(1, n_x):
            x = x_values[j]

            uxx = (u_cur[j + 1] - 2 * u_cur[j] + u_cur[j - 1]) / h**2
            ux = (u_cur[j + 1] - u_cur[j - 1]) / (2 * h)
            ut = (u_cur[j] - u_prev[j]) / tau

            rhs = A_SQ * uxx + B * ux + C * u_cur[j] - D * ut + f(x, t_cur)
            u_new[j] = 2 * u_cur[j] - u_prev[j] + tau**2 * rhs

        if boundary_kind == 1:
            set_boundary_first_order(u_new, t_new, h)
        elif boundary_kind == 2:
            set_boundary_three_point(u_new, t_new, h)
        else:
            set_boundary_taylor_explicit(
                u_new, u_cur, u_prev, t_new, h, tau
            )

        u_prev, u_cur = u_cur, u_new
        layers.append(u_cur)

    return x_values, tau, layers


def solve_implicit(n_x, tau, init_order, boundary_kind):
    # Неявная схема (5.39): производные по x на слое k+1,
    # член d u_t аппроксимируем (u^{k+1} - u^k) / tau
    # (сохраняет трехдиагональность системы, абсолютно устойчива).
    # Внутренние узлы (все производные по x на слое k+1):
    # (u^{k+1} - 2u^k + u^{k-1})/tau^2 = a^2 u_xx^{k+1} + b u_x^{k+1}
    #     + c u^{k+1} - d (u^{k+1} - u^k)/tau + f^{k+1}.
    h = L / n_x
    x_values = get_grid_x(n_x)

    n_t, tau = get_time_steps(tau)

    u_prev = [psi1(x) for x in x_values]
    u_cur = first_layer(x_values, tau, init_order)

    layers = [u_prev, u_cur]
    n = n_x + 1

    for k in range(1, n_t):
        t_new = (k + 1) * tau

        lower = [0.0 for _ in range(n)]
        middle = [0.0 for _ in range(n)]
        upper = [0.0 for _ in range(n)]
        right = [0.0 for _ in range(n)]

        for j in range(1, n_x):
            x = x_values[j]

            lower[j] = -(A_SQ / h**2 - B / (2 * h)) * tau**2
            middle[j] = 1 + 2 * A_SQ * tau**2 / h**2 - C * tau**2 + D * tau
            upper[j] = -(A_SQ / h**2 + B / (2 * h)) * tau**2
            right[j] = (
                2 * u_cur[j]
                - u_prev[j]
                + tau**2 * f(x, t_new)
                + D * tau * u_cur[j]
            )

        phi0 = phi_left(t_new)
        phiN = phi_right(t_new)

        if boundary_kind == 1:
            # (u_1^{k+1} - u_0^{k+1}) / h = phi0.
            middle[0] = 1.0
            upper[0] = -1.0
            right[0] = -h * phi0

            lower[n - 1] = -1.0
            middle[n - 1] = 1.0
            right[n - 1] = h * phiN
        elif boundary_kind == 2:
            # Трехточечные формулы связывают u_0 с u_1, u_2, поэтому
            # исключаем u_0 и u_N из строк j=1 и j=N-1 и решаем
            # уменьшенную систему для u_1..u_{N-1} прогонкой.
            set_implicit_three_point(
                lower, middle, upper, right,
                n_x, u_cur, u_prev, phi0, phiN, h, tau, t_new
            )
        else:
            # Двухточечная 2-й порядок (метод Тейлора): граничное уравнение
            # содержит u_0^{k+1}, u_1^{k+1} и значения с нижних слоев,
            # вписывается в трехдиагональную систему напрямую.
            set_implicit_taylor_rows(
                lower, middle, upper, right,
                n, u_cur, u_prev, phi0, phiN, h, tau, t_new
            )

        if boundary_kind == 2:
            u_new = solve_reduced_system(
                lower, middle, upper, right, n_x
            )

            u_new[0] = (4 * u_new[1] - u_new[2] - 2 * h * phi0) / 3
            u_new[n_x] = (
                4 * u_new[n_x - 1] - u_new[n_x - 2] + 2 * h * phiN
            ) / 3
        else:
            u_new = tridiagonal_solve(lower, middle, upper, right)

        u_prev, u_cur = u_cur, u_new
        layers.append(u_cur)

    return x_values, tau, layers


def set_implicit_three_point(
    lower, middle, upper, right,
    n_x, u_cur, u_prev, phi0, phiN, h, tau, t_new
):
    # Подготовка строк системы: u_0 = (4u_1 - u_2 - 2h phi0)/3,
    # u_N = (4u_{N-1} - u_{N-2} + 2h phiN)/3. Подстановка в строку j=1:
    # (b_1 + 4a_1/3) u_1 + (c_1 - a_1/3) u_2 = r_1 + 2 a_1 h phi0 / 3,
    # в строку j=N-1: (a_{N-1} - c_{N-1}/3) u_{N-2} +
    # (b_{N-1} + 4c_{N-1}/3) u_{N-1} = r_{N-1} - 2 c_{N-1} h phiN / 3.
    a1 = lower[1]
    b1 = middle[1]
    c1 = upper[1]
    r1 = right[1]

    middle[1] = b1 + 4 * a1 / 3
    upper[1] = c1 - a1 / 3
    right[1] = r1 + 2 * a1 * h * phi0 / 3

    a_n = lower[n_x - 1]
    b_n = middle[n_x - 1]
    c_n = upper[n_x - 1]
    r_n = right[n_x - 1]

    lower[n_x - 1] = a_n - c_n / 3
    middle[n_x - 1] = b_n + 4 * c_n / 3
    right[n_x - 1] = r_n - 2 * c_n * h * phiN / 3


def solve_reduced_system(lower, middle, upper, right, n_x):
    # Прогонка уменьшенной системы для неизвестных u_1..u_{N-1}.
    m = n_x - 1

    lo = [0.0 for _ in range(m)]
    mi = [0.0 for _ in range(m)]
    up = [0.0 for _ in range(m)]
    ri = [0.0 for _ in range(m)]

    for j in range(1, n_x):
        lo[j - 1] = lower[j]
        mi[j - 1] = middle[j]
        up[j - 1] = upper[j] if j < n_x - 1 else 0.0
        ri[j - 1] = right[j]

    inner = tridiagonal_solve(lo, mi, up, ri)

    result = [0.0 for _ in range(n_x + 1)]

    for j in range(1, n_x):
        result[j] = inner[j - 1]

    return result


def set_implicit_taylor_rows(
    lower, middle, upper, right,
    n, u_cur, u_prev, phi0, phiN, h, tau, t_new
):
    # Граничные уравнения Тейлора для неявной схемы.
    # Из разложения u_1 = u_0 + h u_x(0) + h^2/2 u_xx(0) с заменой u_xx
    # из ДУ (u_xx = u_tt + d u_t - b u_x - c u - f, все на слое k+1,
    # u_tt и u_t - разности по времени в узле 0):
    # (1 + h^2/(2tau^2) + d h^2/(2tau) - c h^2/2) u_0^{k+1} - u_1^{k+1}
    #   = -h phi0 + b h^2/2 phi0 + h^2/2 f(0, t)
    #     + h^2/(2tau^2)(2 u_0^k - u_0^{k-1}) + d h^2/(2tau) u_0^k.
    half = h * h / 2

    aa = 1 + half / tau**2 + half * D / tau - half * C

    if abs(aa) < 1e-14:
        raise ValueError("Вырожденное граничное уравнение Тейлора")

    middle[0] = aa
    upper[0] = -1.0
    right[0] = (
        -h * phi0
        + half * B * phi0
        + half * f(0.0, t_new)
        + half * (2 * u_cur[0] - u_prev[0]) / tau**2
        + half * D * u_cur[0] / tau
    )

    middle[n - 1] = aa
    lower[n - 1] = -1.0
    right[n - 1] = (
        h * phiN
        + half * B * phiN
        + half * f(L, t_new)
        + half * (2 * u_cur[n - 1] - u_prev[n - 1]) / tau**2
        + half * D * u_cur[n - 1] / tau
    )


def compute_errors(x_values, tau, layers, times):
    # Погрешность в моменты times (ближайшие слои).
    result = []

    for t in times:
        k = int(round(t / tau))

        if k < 0 or k >= len(layers):
            raise ValueError(
                f"Момент времени t = {t} вне расчетного диапазона"
            )

        result.append(get_max_error(layers[k], x_values, t))

    return result


def convergence_rows(solver, boundary_kind, init_order, n_x_list, tau_rule):
    # Погрешность при t = T на последовательности сеток.
    rows = []
    prev = None

    for n_x in n_x_list:
        h = L / n_x
        x_values, tau, layers = solver(
            n_x, tau_rule(n_x), init_order, boundary_kind
        )
        error = get_max_error(layers[-1], x_values, T)

        ratio = None if prev is None else prev / error
        rows.append((h, tau, error, ratio))
        prev = error

    return rows


def estimate_order(rows):
    # Порядок сходимости по двум последним строкам.
    (_, _, e1, _), (_, _, e2, _) = rows[-2], rows[-1]
    return math.log(e1 / e2) / math.log(2)


def main():
    n_x = 50
    h = L / n_x
    tau = 0.5 * h

    times = [0.1, 0.5, 1.0]

    print("Лабораторная работа 6, вариант 10")
    print("u_tt + 3 u_t = u_xx + u_x - u - cos(x) exp(-t),  0 < x < pi")
    print("u_x(0, t) = exp(-t),  u_x(pi, t) = -exp(-t)")
    print("u(x, 0) = sin(x),  u_t(x, 0) = -sin(x)")
    print("Точное решение: U(x, t) = exp(-t) sin(x)")
    print()

    residual = check_source_consistency()
    print("Проверка согласованности источника и точного решения:")
    print(f"max невязки (u_tt + 3u_t) - (u_xx + u_x - u + f) = {residual:.2e}")
    print()

    # Основные расчеты: обе схемы, 2-й порядок начального условия,
    # три варианта гранусловий.
    for scheme_name, solver in (
        ("Явная схема крест", solve_explicit),
        ("Неявная схема", solve_implicit),
    ):
        for boundary_kind, boundary_name in enumerate(BOUNDARY_NAMES, 1):
            x_values, tau_used, layers = solver(
                n_x, tau, init_order=2, boundary_kind=boundary_kind
            )

            print(f"{scheme_name}, гранусловия: {boundary_name}")
            print(f"h = {h:.6f}, tau = {tau_used:.6f}")

            if solver is solve_explicit:
                sigma = tau_used**2 / h**2
                print(f"sigma = tau^2/h^2 = {sigma:.4f} < 1")

            errors = compute_errors(x_values, tau_used, layers, times)
            print_error_summary("Погрешность численного решения:", times, errors)

    # Сравнение порядков аппроксимации второго начального условия.
    for scheme_name, solver in (
        ("Явная схема крест", solve_explicit),
        ("Неявная схема", solve_implicit),
    ):
        print(
            f"{scheme_name}, сравнение порядков начального условия"
        )
        print("(гранусловия трехточечные 2-й порядок)")
        print(f"h = {h:.6f}, tau = {tau:.6f}")

        for order in (1, 2):
            x_values, tau_used, layers = solver(
                n_x, tau, init_order=order, boundary_kind=2
            )
            errors = compute_errors(x_values, tau_used, layers, times)
            print_error_summary(
                f"Начальное условие {order}-го порядка:", times, errors
            )

    # Таблица погрешности на слое t = T.
    x_values, tau_used, layers = solve_implicit(
        n_x, tau, init_order=2, boundary_kind=2
    )
    print_layer_table(
        "Неявная схема (гранусловия трехточечные), слой t = T:",
        layers[-1],
        x_values,
        T,
    )

    # Исследование зависимости погрешности от h при tau = h/4.
    n_x_list = [10, 20, 40, 80]

    print("Зависимость погрешности от h (tau = h/4, t = T)")
    print("Член 3u_t аппроксимирован односторонней разностью,")
    print("поэтому погрешность ~ O(tau) + O(h^2) и порядок по h ~ 1")

    for scheme_name, solver in (
        ("Явная схема крест", solve_explicit),
        ("Неявная схема", solve_implicit),
    ):
        for boundary_kind, boundary_name in enumerate(BOUNDARY_NAMES, 1):
            rows = convergence_rows(
                solver,
                boundary_kind,
                init_order=2,
                n_x_list=n_x_list,
                tau_rule=lambda n: (L / n) / 4,
            )
            print_convergence_table(
                f"{scheme_name}, гранусловия: {boundary_name}", rows
            )

    # Исследование зависимости от tau при фиксированном h.
    print("Зависимость погрешности от tau (h = pi/50, t = T)")
    print("Явная схема крест, гранусловия трехточечные, нач.усл. 2-й порядок")

    rows = []
    prev = None

    for tau_value in [0.01, 0.005, 0.0025, 0.00125]:
        x_values, tau_used, layers = solve_explicit(
            n_x, tau_value, init_order=2, boundary_kind=2
        )
        error = get_max_error(layers[-1], x_values, T)
        ratio = None if prev is None else prev / error
        rows.append((h, tau_used, error, ratio))
        prev = error

    print_convergence_table("", rows)

    # Порядки сходимости по h при tau = h^2/20 (tau-погрешность пренебрежима).
    print("Порядок сходимости по h при tau = h^2/20 (t = T)")
    print(" tau-погрешность мала, виден порядок аппроксимации по x")

    order_rows = []

    for scheme_name, solver in (
        ("явная", solve_explicit),
        ("неявная", solve_implicit),
    ):
        for init_order in (1, 2):
            for boundary_kind, boundary_name in enumerate(
                BOUNDARY_NAMES, 1
            ):
                rows = convergence_rows(
                    solver,
                    boundary_kind,
                    init_order=init_order,
                    n_x_list=[20, 40, 80],
                    tau_rule=lambda n: (L / n) ** 2 / 20,
                )
                order_rows.append(
                    (
                        scheme_name,
                        init_order,
                        boundary_name,
                        estimate_order(rows),
                    )
                )

    print_orders_table("", order_rows)

    # Порядок сходимости по tau при фиксированном h (обе схемы).
    print("Порядок сходимости по tau (h = pi/200, t = T)")
    print("гранусловия трехточечные, нач.усл. 2-й порядок")

    n_fixed = 200

    for scheme_name, solver in (
        ("Явная схема крест", solve_explicit),
        ("Неявная схема", solve_implicit),
    ):
        rows = []
        prev = None

        for tau_value in [0.01, 0.005, 0.0025, 0.00125]:
            x_values, tau_used, layers = solver(
                n_fixed, tau_value, init_order=2, boundary_kind=2
            )
            error = get_max_error(layers[-1], x_values, T)
            ratio = None if prev is None else prev / error
            rows.append((L / n_fixed, tau_used, error, ratio))
            prev = error

        print_convergence_table(scheme_name, rows)


if __name__ == "__main__":
    main()
