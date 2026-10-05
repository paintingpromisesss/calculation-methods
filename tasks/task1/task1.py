import math

from utils import (
    A_COEF,
    B_COEF,
    C_COEF,
    L,
    T,
    exact_solution,
    phi_left,
    phi_right,
    get_grid_x,
    get_tau,
    get_max_error,
    print_solution_table,
    print_grid_error_table,
    print_order_table,
    tridiagonal_solve,
)


# Схема с весами (5.20) для уравнения u_t = a^2 u_xx + b u_x + c u:
# (u^{k+1} - u^k)/tau =
#     theta * (a^2 Lh + b Dh + c)(u^{k+1}) + (1 - theta) * (a^2 Lh + b Dh + c)(u^k),
# где Lh - вторая разность, Dh - центральная разность по x.
# theta = 0 - явная схема, theta = 1 - неявная, theta = 0.5 - Кранк-Николсон.

# Способы аппроксимации граничных условий 3-го рода u_x + u = phi:
BC_TWO_POINT_1 = "two_point_1"    # двухточечная, 1-й порядок
BC_THREE_POINT_2 = "three_point_2"  # трехточечная, 2-й порядок
BC_TAYLOR_2 = "taylor_2"          # двухточечная, 2-й порядок (ряд Тейлора)


def solve_layer_weighted(theta, u, x_values, h, tau, t_new, bc_method):
    """Один шаг по времени схемы с весами theta.

    Возвращает слой u^{k+1}. Граничные условия задает bc_method.
    """
    a2 = A_COEF * A_COEF
    b = B_COEF
    c = C_COEF

    n = len(x_values) - 1

    sigma = a2 * tau / h**2

    # Явная часть условно устойчива: sigma <= 1/2 при theta < 1/2
    if theta < 0.5 - 1e-12 and sigma > 0.5 + 1e-12:
        raise ValueError("Нарушено условие устойчивости явной части: sigma <= 1/2")

    # Неизвестные u^{k+1}, матрица системы (a_j, b_j, c_j) и правая часть d_j
    lower = [0.0 for _ in range(n + 1)]
    middle = [0.0 for _ in range(n + 1)]
    upper = [0.0 for _ in range(n + 1)]
    right = [0.0 for _ in range(n + 1)]

    # Внутренние узлы j = 1, N-1:
    # u_j^{k+1} - theta*tau*(a^2 Lh + b Dh + c)u^{k+1}_j
    #     = u_j^k + (1 - theta)*tau*(a^2 Lh + b Dh + c)u^k_j
    for j in range(1, n):
        lx_explicit = (
            (1 - theta) * tau * a2 * (u[j + 1] - 2 * u[j] + u[j - 1]) / h**2
            + (1 - theta) * tau * b * (u[j + 1] - u[j - 1]) / (2 * h)
            + (1 - theta) * tau * c * u[j]
        )

        right[j] = u[j] + lx_explicit

        lower[j] = -theta * tau * (a2 / h**2 - b / (2 * h))
        middle[j] = 1 + theta * tau * (2 * a2 / h**2 - c)
        upper[j] = -theta * tau * (a2 / h**2 + b / (2 * h))

    if bc_method == BC_TWO_POINT_1:
        # Двухточечная аппроксимация 1-го порядка (направленные разности):
        # (u_1 - u_0)/h + u_0 = phi_0(t),  (u_N - u_{N-1})/h + u_N = phi_l(t)
        middle[0] = 1.0 - 1.0 / h
        upper[0] = 1.0 / h
        right[0] = phi_left(t_new)

        lower[n] = -1.0 / h
        middle[n] = 1.0 + 1.0 / h
        right[n] = phi_right(t_new)

    elif bc_method == BC_THREE_POINT_2:
        # Трехточечная аппроксимация 2-го порядка:
        # (-3u_0 + 4u_1 - u_2)/(2h) + u_0 = phi_0(t),
        # (u_{N-2} - 4u_{N-1} + 3u_N)/(2h) + u_N = phi_l(t).
        # СЛАУ теряет трехдиагональность (граничные строки содержат три
        # неизвестных). По 5.1.3 исключаем u_2 из левого граничного уравнения
        # линейной комбинацией с уравнением узла j = 1 (и u_{N-2} из правого
        # с уравнением j = N-1), сохраняя применимость метода прогонки.

        # Левое граничное: (1 - 3/(2h))u_0 + (2/h)u_1 - (1/(2h))u_2 = phi_0
        # Исключаем u_2: row0 - (coef_u2 / upper[1]) * row1
        coef_u2 = -1.0 / (2 * h)

        if abs(upper[1]) < 1e-30:
            raise ValueError("Деление на ноль при исключении u_2")

        factor = coef_u2 / upper[1]

        middle[0] = (1.0 - 3.0 / (2 * h)) - factor * lower[1]
        upper[0] = (2.0 / h) - factor * middle[1]
        # коэффициент при u_2: coef_u2 - factor * upper[1] = 0
        right[0] = phi_left(t_new) - factor * right[1]

        # Правое граничное: (1/(2h))u_{N-2} + (-2/h)u_{N-1} + (1 + 3/(2h))u_N = phi_l
        # Исключаем u_{N-2}: rowN - (coef_uNm2 / lower[n-1]) * row_{n-1}
        coef_uNm2 = 1.0 / (2 * h)

        if abs(lower[n - 1]) < 1e-30:
            raise ValueError("Деление на ноль при исключении u_{N-2}")

        factor_r = coef_uNm2 / lower[n - 1]

        lower[n] = (-2.0 / h) - factor_r * middle[n - 1]
        middle[n] = (1.0 + 3.0 / (2 * h)) - factor_r * upper[n - 1]
        right[n] = phi_right(t_new) - factor_r * right[n - 1]

    else:
        # Двухточечная аппроксимация 2-го порядка по методу рядов Тейлора
        # (5.26)-(5.29): разложение u_1 в ряд Тейлора около x=0,
        # подстановка u_xx из уравнения u_xx = (u_t - b u_x - c u)/a^2,
        # исключение u_x через граничное условие u_x(0,t) = phi_0 - u_0.
        b0, c0, d0 = taylor_boundary_left(h, tau, u[0], t_new)
        middle[0] = b0
        upper[0] = c0
        right[0] = d0

        aN, bN, dN = taylor_boundary_right(h, tau, u[n], t_new)
        lower[n] = aN
        middle[n] = bN
        right[n] = dN

    return tridiagonal_solve(lower, middle, upper, right)


def solve_layer_explicit_three_point(u, x_values, h, tau, t_new):
    """Явная схема с трехточечной аппроксимацией граничных условий.

    Внутренние узлы вычисляются явно (theta = 0), затем граничные значения
    находятся из трехточечных формул численного дифференцирования 2-го
    порядка: (-3u_0 + 4u_1 - u_2)/(2h) + u_0 = phi_0(t) и
    (u_{N-2} - 4u_{N-1} + 3u_N)/(2h) + u_N = phi_l(t).
    """
    a2 = A_COEF * A_COEF
    b = B_COEF
    c = C_COEF

    n = len(x_values) - 1

    sigma = a2 * tau / h**2

    if sigma > 0.5 + 1e-12:
        raise ValueError("Нарушено условие устойчивости явной схемы: sigma <= 1/2")

    u_new = [0.0 for _ in range(n + 1)]

    # Явный шаг во внутренних узлах
    for j in range(1, n):
        u_new[j] = u[j] + tau * (
            a2 * (u[j + 1] - 2 * u[j] + u[j - 1]) / h**2
            + b * (u[j + 1] - u[j - 1]) / (2 * h)
            + c * u[j]
        )

    # Левая граница: (1 - 3/(2h))u_0 = phi_0 - (4u_1 - u_2)/(2h)
    denom_left = 1.0 - 3.0 / (2 * h)
    if abs(denom_left) < 1e-30:
        raise ValueError("Вырожденное граничное уравнение на левой границе")

    u_new[0] = (
        phi_left(t_new) - (4 * u_new[1] - u_new[2]) / (2 * h)
    ) / denom_left

    # Правая граница: (1 + 3/(2h))u_N = phi_l - (u_{N-2} - 4u_{N-1})/(2h)
    denom_right = 1.0 + 3.0 / (2 * h)
    if abs(denom_right) < 1e-30:
        raise ValueError("Вырожденное граничное уравнение на правой границе")

    u_new[n] = (
        phi_right(t_new) - (u_new[n - 2] - 4 * u_new[n - 1]) / (2 * h)
    ) / denom_right

    return u_new


def taylor_boundary_left(h, tau, u0_old, t_new):
    """Коэффициенты (b0, c0, d0) уравнения (5.28): b0*u_0 + c0*u_1 = d0.

    Вывод: u_1 = u_0 + h*u_x + h^2/2*u_xx + O(h^3),
    u_xx = (u_t - b*u_x - c*u)/a^2 (из уравнения),
    u_x(0,t) = phi_0(t) - u_0 (из граничного условия u_x + u = phi_0),
    u_t ~ (u_0^{k+1} - u_0^k)/tau + O(tau).
    Итог: u_1 = coef0*u_0^{k+1} + C, т.е. coef0*u_0 - u_1 = -C,
    где coef0 = 1 - h + h^2/(2a^2)(1/tau + b - c),
    C = h*phi_0 - h^2*b/(2a^2)*phi_0 - h^2/(2a^2)*u_0^k/tau.
    """
    a2 = A_COEF * A_COEF
    b = B_COEF
    c = C_COEF

    phi = phi_left(t_new)

    coef0 = 1.0 - h + (h**2 / (2 * a2)) * (1 / tau + b - c)
    c_const = h * phi - (h**2 * b / (2 * a2)) * phi - (h**2 / (2 * a2)) * u0_old / tau

    return coef0, -1.0, -c_const


def taylor_boundary_right(h, tau, uN_old, t_new):
    """Коэффициенты (aN, bN, dN) уравнения (5.29): aN*u_{N-1} + bN*u_N = dN.

    Аналогично левой границе, но u_{N-1} = u_N - h*u_x + h^2/2*u_xx
    и u_x(l,t) = phi_l(t) - u_N.
    """
    a2 = A_COEF * A_COEF
    b = B_COEF
    c = C_COEF

    phi = phi_right(t_new)

    coefN = 1.0 + h + (h**2 / (2 * a2)) * (1 / tau + b - c)
    c_const = -h * phi - (h**2 * b / (2 * a2)) * phi - (h**2 / (2 * a2)) * uN_old / tau

    return -1.0, coefN, -c_const


def solve_scheme(theta, n, k_total, bc_method):
    """Полный расчет по схеме с весами theta.

    Возвращает (layers, x_values, h, tau), layers[k] - слой в момент t=k*tau.
    """
    h = L / n
    tau = get_tau(k_total)

    x_values = get_grid_x(n)

    # Начальный слой из начального условия u(x,0) = sin(x)
    u = [math.sin(x) for x in x_values]

    layers = [list(u)]

    for k in range(k_total):
        t_new = (k + 1) * tau
        if theta == 0.0 and bc_method == BC_THREE_POINT_2:
            # Явная схема: граничные условия 2-го порядка применяются
            # после явного вычисления внутренних узлов
            u = solve_layer_explicit_three_point(u, x_values, h, tau, t_new)
        else:
            u = solve_layer_weighted(theta, u, x_values, h, tau, t_new, bc_method)
        layers.append(list(u))

    return layers, x_values, h, tau


def main():
    n = 50
    k_total = 1000

    theta_explicit = 0.0
    theta_implicit = 1.0
    theta_crank = 0.5

    schemes = [
        ("Явная схема (theta = 0)", theta_explicit),
        ("Неявная схема (theta = 1)", theta_implicit),
        ("Схема Кранка-Николсона (theta = 0.5)", theta_crank),
    ]

    bc_methods = [
        ("Двухточечная 1-го порядка", BC_TWO_POINT_1),
        ("Трехточечная 2-го порядка", BC_THREE_POINT_2),
        ("Двухточечная 2-го порядка (Тейлор)", BC_TAYLOR_2),
    ]

    t_check = [0.1, 0.5, 1.0]

    # Сетки для исследования сходимости: одновременно измельчаем h и tau
    # K ~ N^2, т.е. tau ~ h^2, чтобы погрешность по времени не маскировала
    # погрешность по пространству
    grid_params = [(20, 200), (40, 800), (80, 3200), (160, 12800)]

    print("Задача: u_t = a^2 u_xx + b u_x + c u, a = 1, b = 1, c = -1")
    print("u_x(0,t) + u(0,t) = phi_0(t), u_x(pi,t) + u(pi,t) = -phi_0(t)")
    print("u(x,0) = sin(x), 0 <= x <= pi, 0 <= t <= 1")
    print("Аналитическое решение: U = exp((c-a)t) sin(x + bt)")
    print(f"N = {n}, K = {k_total}")
    print()

    for scheme_name, theta in schemes:
        for bc_name, bc_method in bc_methods:
            print(f"{scheme_name}, граничные условия: {bc_name}")

            layers, x_values, h, tau = solve_scheme(
                theta, n, k_total, bc_method
            )

            for t in t_check:
                k = round(t / tau)
                print_solution_table(
                    f"t = {t:.2f}:",
                    x_values,
                    layers[k],
                    k * tau
                )

    # Исследование зависимости погрешности от сеточных параметров tau и h
    for scheme_name, theta in schemes:
        for bc_name, bc_method in bc_methods:
            errors = []

            for n_i, k_i in grid_params:
                layers, x_values, h, tau = solve_scheme(
                    theta, n_i, k_i, bc_method
                )
                errors.append(get_max_error(layers[-1], x_values, T))

            # Ожидаемый порядок: 1 при гранусловиях 1-го порядка,
            # 2 при гранусловиях 2-го порядка (tau достаточно мал)
            expected = 1 if bc_method == BC_TWO_POINT_1 else 2

            print_grid_error_table(
                f"Сходимость: {scheme_name}, {bc_name}:",
                grid_params,
                errors
            )

            print_order_table(
                f"Порядок сходимости: {scheme_name}, {bc_name}:",
                grid_params,
                errors,
                expected
            )


if __name__ == "__main__":
    main()
