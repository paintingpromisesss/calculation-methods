from utils import (
    f,
    g,
    get_grid,
    get_errors,
    runge_romberg_error,
    print_method_table,
    print_runge_romberg_table,
)


def euler_method(a, b, h, y0, z0):
    points = []
    x_values = get_grid(a, b, h)

    y = y0
    z = z0

    points.append((x_values[0], y, z))

    for k in range(len(x_values) - 1):
        x = x_values[k]

        y_next = y + h * f(x, y, z)
        z_next = z + h * g(x, y, z)

        y = y_next
        z = z_next

        points.append((x_values[k + 1], y, z))

    return points


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


def adams_method(a, b, h, y0, z0):
    x_values = get_grid(a, b, h)

    if len(x_values) < 4:
        raise ValueError("Для метода Адамса нужно минимум 4 узла")

    start_points = runge_kutta_method(a, a + 3 * h, h, y0, z0)
    points = start_points[:]

    for k in range(3, len(x_values) - 1):
        x_k, y_k, z_k = points[k]

        x_k1, y_k1, z_k1 = points[k - 1]
        x_k2, y_k2, z_k2 = points[k - 2]
        x_k3, y_k3, z_k3 = points[k - 3]

        f_k = f(x_k, y_k, z_k)
        f_k1 = f(x_k1, y_k1, z_k1)
        f_k2 = f(x_k2, y_k2, z_k2)
        f_k3 = f(x_k3, y_k3, z_k3)

        g_k = g(x_k, y_k, z_k)
        g_k1 = g(x_k1, y_k1, z_k1)
        g_k2 = g(x_k2, y_k2, z_k2)
        g_k3 = g(x_k3, y_k3, z_k3)

        y_next = y_k + h / 24 * (
            55 * f_k - 59 * f_k1 + 37 * f_k2 - 9 * f_k3
        )

        z_next = z_k + h / 24 * (
            55 * g_k - 59 * g_k1 + 37 * g_k2 - 9 * g_k3
        )

        points.append((x_values[k + 1], y_next, z_next))

    return points


def main():
    a = 1.0
    b = 2.0
    h = 0.1

    y0 = 3.0
    z0 = 2.0

    euler_points = euler_method(a, b, h, y0, z0)
    runge_kutta_points = runge_kutta_method(a, b, h, y0, z0)
    adams_points = adams_method(a, b, h, y0, z0)

    print_method_table("Метод Эйлера:", euler_points)
    print_method_table("Метод Рунге-Кутты 4-го порядка:", runge_kutta_points)
    print_method_table("Метод Адамса 4-го порядка:", adams_points)

    h2 = 2 * h

    euler_points_h2 = euler_method(a, b, h2, y0, z0)
    runge_kutta_points_h2 = runge_kutta_method(a, b, h2, y0, z0)
    adams_points_h2 = adams_method(a, b, h2, y0, z0)

    euler_rr = runge_romberg_error(euler_points, euler_points_h2, p=1)
    runge_kutta_rr = runge_romberg_error(
        runge_kutta_points,
        runge_kutta_points_h2,
        p=4
    )
    adams_rr = runge_romberg_error(adams_points, adams_points_h2, p=4)

    print("Оценка погрешности методом Рунге-Ромберга:")
    print()

    print_runge_romberg_table(
        "Метод Эйлера, p = 1:",
        euler_rr
    )

    print_runge_romberg_table(
        "Метод Рунге-Кутты 4-го порядка, p = 4:",
        runge_kutta_rr
    )

    print_runge_romberg_table(
        "Метод Адамса 4-го порядка, p = 4:",
        adams_rr
    )

    print("Сравнение максимальных погрешностей по точному решению:")
    print(f"Метод Эйлера: {max(error for x, y, z, y_exact, error in get_errors(euler_points)):.10f}")
    print(f"Метод Рунге-Кутты 4-го порядка: {max(error for x, y, z, y_exact, error in get_errors(runge_kutta_points)):.10f}")
    print(f"Метод Адамса 4-го порядка: {max(error for x, y, z, y_exact, error in get_errors(adams_points)):.10f}")


if __name__ == "__main__":
    main()