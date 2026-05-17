from utils import (
    EPS,
    f,
    g,
    p,
    q,
    get_grid,
    get_max_error,
    get_max_finite_difference_error,
    runge_romberg_error,
    print_shooting_iterations_table,
    print_shooting_table,
    print_finite_difference_table,
    print_runge_romberg_table,
    tridiagonal_solve,
)


def runge_kutta_method(a, b, h, y0, z0):
    points = []
    x_values = get_grid(a, b, h)

    y = y0
    z = z0

    points.append((x_values[0], y, z))

    for k in range(len(x_values) - 1):
        x = x_values[k]

        k1 = h * f(x, y, z)
        l1 = h * g(x, y, z)

        k2 = h * f(
            x + h / 2,
            y + k1 / 2,
            z + l1 / 2
        )
        l2 = h * g(
            x + h / 2,
            y + k1 / 2,
            z + l1 / 2
        )

        k3 = h * f(
            x + h / 2,
            y + k2 / 2,
            z + l2 / 2
        )
        l3 = h * g(
            x + h / 2,
            y + k2 / 2,
            z + l2 / 2
        )

        k4 = h * f(
            x + h,
            y + k3,
            z + l3
        )
        l4 = h * g(
            x + h,
            y + k3,
            z + l3
        )

        delta_y = (k1 + 2 * k2 + 2 * k3 + k4) / 6
        delta_z = (l1 + 2 * l2 + 2 * l3 + l4) / 6

        y += delta_y
        z += delta_z

        points.append((x_values[k + 1], y, z))

    return points


def shooting_residual(points):
    x, y, z = points[-1]
    return y - z - 1


def shooting_method(a, b, h, eta0, eta1, eps=1e-7, max_iter=100):
    iterations = []

    z0 = 0.0

    points0 = runge_kutta_method(a, b, h, eta0, z0)
    phi0 = shooting_residual(points0)

    iterations.append((0, eta0, points0[-1][1], points0[-1][2], phi0))

    points1 = runge_kutta_method(a, b, h, eta1, z0)
    phi1 = shooting_residual(points1)

    iterations.append((1, eta1, points1[-1][1], points1[-1][2], phi1))

    if abs(phi0) <= eps:
        return points0, eta0, iterations

    if abs(phi1) <= eps:
        return points1, eta1, iterations

    for j in range(2, max_iter + 1):
        if abs(phi1 - phi0) < EPS:
            raise ValueError("Метод секущих не может продолжаться: деление на ноль")

        eta2 = eta1 - phi1 * (eta1 - eta0) / (phi1 - phi0)

        points2 = runge_kutta_method(a, b, h, eta2, z0)
        phi2 = shooting_residual(points2)

        iterations.append((j, eta2, points2[-1][1], points2[-1][2], phi2))

        if abs(phi2) <= eps:
            return points2, eta2, iterations

        eta0 = eta1
        phi0 = phi1

        eta1 = eta2
        phi1 = phi2

    raise ValueError("Метод стрельбы не сошелся за заданное число итераций")



# Используем односторонние разности первого порядка на границах.
# По методичке в этом случае сохраняется трехдиагональная система,
# поэтому можно применять метод прогонки, но общий порядок схемы p = 1.

def finite_difference_method(a, b, h):
    x_values = get_grid(a, b, h)
    n = len(x_values) - 1

    if n < 2:
        raise ValueError("Для конечно-разностного метода нужно минимум 3 узла")

    lower = [0.0 for _ in range(n + 1)]
    middle = [0.0 for _ in range(n + 1)]
    upper = [0.0 for _ in range(n + 1)]
    right = [0.0 for _ in range(n + 1)]

    # y'(0) = 0
    # (y1 - y0) / h = 0
    middle[0] = -1.0
    upper[0] = 1.0
    right[0] = 0.0

    # Внутренние узлы:
    # y'' + p(x)y' + q(x)y = f(x)
    for k in range(1, n):
        x = x_values[k]

        p_k = p(x)
        q_k = q(x)

        lower[k] = 1 - p_k * h / 2
        middle[k] = -2 + q_k * h**2
        upper[k] = 1 + p_k * h / 2
        right[k] = 0.0

    # y(2) - y'(2) = 1
    # y'(2) ~= (yN - yN-1) / h
    # yN - (yN - yN-1) / h = 1
    lower[n] = 1 / h
    middle[n] = 1 - 1 / h
    upper[n] = 0.0
    right[n] = 1.0

    y_values = tridiagonal_solve(lower, middle, upper, right)

    return [
        (x_values[i], y_values[i])
        for i in range(n + 1)
    ]


def main():
    a = 0.0
    b = 2.0
    h = 0.1
    h2 = 2 * h

    eta0 = 0.0
    eta1 = 2.0

    print("(x^2 + 1)y'' - 2y = 0")
    print("y'(0) = 0")
    print("y(2) - y'(2) = 1")
    print(f"x принадлежит [{a:.1f}, {b:.1f}], h = {h:.1f}")
    print("Точное решение:")
    print("y = x^2 + 1")
    print()

    print("После замены z = y' получаем систему:")
    print("y' = z")
    print("z' = 2y / (x^2 + 1)")
    print()

    print("Метод стрельбы")
    print("Пусть y(0) = eta, y'(0) = 0")
    print()

    shooting_points, eta, shooting_iterations = shooting_method(
        a, b, h, eta0, eta1
    )

    shooting_points_h2, eta_h2, shooting_iterations_h2 = shooting_method(
        a, b, h2, eta0, eta1
    )

    print_shooting_iterations_table(
        "Итерации метода секущих для метода стрельбы:",
        shooting_iterations
    )

    print(f"Найденное eta при h = {h:.10f}: {eta:.10f}")
    print()

    print_shooting_table(
        "Метод стрельбы:",
        shooting_points
    )

    print("Конечно-разностный метод")
    print("Приводим уравнение к виду y'' + p(x)y' + q(x)y = f(x)")
    print("p(x) = 0")
    print("q(x) = -2 / (x^2 + 1)")
    print("f(x) = 0")
    print()

    finite_difference_points = finite_difference_method(a, b, h)
    finite_difference_points_h2 = finite_difference_method(a, b, h2)

    print_finite_difference_table(
        "Конечно-разностный метод:",
        finite_difference_points
    )

    shooting_rr = runge_romberg_error(
        shooting_points,
        shooting_points_h2,
        p=4,
        is_system=True
    )

    finite_difference_rr = runge_romberg_error(
        finite_difference_points,
        finite_difference_points_h2,
        p=1,
        is_system=False
    )

    print("Оценка погрешности методом Рунге-Ромберга:")
    print()

    print_runge_romberg_table(
        "Метод стрельбы, p = 4:",
        shooting_rr
    )

    print_runge_romberg_table(
        "Конечно-разностный метод, p = 1:",
        finite_difference_rr
    )

    print("Сравнение максимальных погрешностей по точному решению:")
    print(f"Метод стрельбы: {get_max_error(shooting_points):.10f}")
    print(
        "Конечно-разностный метод: "
        f"{get_max_finite_difference_error(finite_difference_points):.10f}"
    )


if __name__ == "__main__":
    main()