import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from utils import (
    A,
    B,
    N_VALUES,
    OMEGA,
    TOL,
    convergence_order,
    get_grid,
    get_max_error,
    initial_guess,
    print_method_table,
    print_order_table,
    print_solution_table,
    set_boundary,
)


MAX_ITER = 100000


def new_grid(n):
    # Построение сетки, начального приближения и граничных условий
    h, xs, ys = get_grid(n)
    u = [[0.0] * (n + 1) for _ in range(n + 1)]
    set_boundary(u, xs, ys, n)
    initial_guess(u, xs, ys, n)
    return h, xs, ys, u


def denom(h):
    # Знаменатель разностно-итерационной формулы
    return 4.0 * (1.0 - h * h)


def next_value(u, i, j, h):
    # Центральная разностная схема для уравнения
    # u_xx + u_yy = -2u_x - 2u_y - 4u:
    # (u_{i+1,j} - 2u_{i,j} + u_{i-1,j} + u_{i,j+1} - 2u_{i,j} + u_{i,j-1}) / h^2
    #   = -2(u_{i+1,j} - u_{i-1,j})/(2h) - 2(u_{i,j+1} - u_{i,j-1})/(2h) - 4u_{i,j}
    # => u_{i,j} = [(1+h)(u_{i+1,j} + u_{i,j+1}) + (1-h)(u_{i-1,j} + u_{i,j-1})]
    #              / (4(1 - h^2))
    return (
        (1 + h) * (u[j][i + 1] + u[j + 1][i])
        + (1 - h) * (u[j][i - 1] + u[j - 1][i])
    ) / denom(h)


def solve(n, method, omega=OMEGA, tol=TOL):
    # Решение СЛАУ итерационным методом.
    # method: "libman" - простые итерации (Либман),
    #         "seidel" - Зейдель,
    #         "relax"  - верхняя релаксация
    if method not in ("libman", "seidel", "relax"):
        raise ValueError("Неизвестный метод: " + method)

    if not 1 < omega < 2:
        raise ValueError("Параметр релаксации должен лежать в (1, 2)")

    h, xs, ys, u = new_grid(n)

    for k in range(1, MAX_ITER + 1):
        if method == "libman":
            # Итерация Либмана: все соседи берутся с предыдущей итерации
            new_u = [row[:] for row in u]
            src, dst = u, new_u
        else:
            # Зейдель и релаксация: пересчитанные узлы используются сразу
            src = dst = u

        max_diff = 0.0
        for j in range(1, n):
            for i in range(1, n):
                val = next_value(src, i, j, h)
                if method == "relax":
                    # Верхняя релаксация: u_new = u_old + w (u_libman - u_old)
                    val = dst[j][i] + omega * (val - dst[j][i])
                diff = abs(val - dst[j][i])
                if diff > max_diff:
                    max_diff = diff
                dst[j][i] = val

        if method == "libman":
            u = new_u

        if max_diff <= tol:
            return u, xs, ys, h, k, max_diff

    raise ValueError(
        "Метод " + method + " не сошелся за " + str(MAX_ITER) + " итераций"
    )


def solve_all_methods(n):
    # Решение тремя методами на сетке с N разбиениями
    results = []
    for method, name in (
        ("libman", "Метод Либмана (простых итераций)"),
        ("seidel", "Метод Зейделя"),
        ("relax", "Метод верхней релаксации (w = " + str(OMEGA) + ")"),
    ):
        u, xs, ys, h, iterations, diff = solve(n, method)
        error = get_max_error(u, xs, ys, n)
        results.append((method, name, u, xs, ys, h, iterations, diff, error))
    return results


def main():
    n_base = 16

    print("Краевая задача: u_xx + u_yy = -2u_x - 2u_y - 4u,")
    print("x, y из [0, pi/2], все границы - условия Дирихле")
    print("Точное решение: U(x, y) = exp(-x-y) cos(x) cos(y)")
    print()

    print("Решение на базовой сетке N = " + str(n_base))
    print()

    base_results = solve_all_methods(n_base)

    for method, name, u, xs, ys, h, iterations, diff, error in base_results:
        print_solution_table(
            name + ", N = " + str(n_base) + ", итераций: " + str(iterations),
            u,
            xs,
            ys,
            n_base
        )

    # Исследование зависимости погрешности от h
    for method, name, _, _, _, _, _, _, _ in base_results:
        rows = []
        order_rows = []
        prev_error = None

        for n in N_VALUES:
            u, xs, ys, h, iterations, diff = solve(n, method)
            error = get_max_error(u, xs, ys, n)

            order = None
            if prev_error is not None:
                order = convergence_order(error, prev_error)

            rows.append((n, h, iterations, diff, error))
            order_rows.append((n, h, error, order))
            prev_error = error

        print_method_table("Исследование сходимости: " + name, rows)
        print_order_table("Порядок сходимости: " + name, order_rows)

    # Сравнение числа итераций методов
    print("Сравнение методов на базовой сетке N = " + str(n_base) + ":")
    print()
    for method, name, u, xs, ys, h, iterations, diff, error in base_results:
        print(
            f"{name:<44}"
            f"итераций: {iterations:>6}"
            f"    погрешность: {error:.3e}"
        )
    print()


if __name__ == "__main__":
    main()
