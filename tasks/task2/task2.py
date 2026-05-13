from utils import (
    EPS,
    validate_values,
    calculate_h_values,
    solve_linear_system,
    find_segment,
    print_values_table,
    print_h_values,
    print_c_system,
    print_coefficients_table,
    print_spline_polynomials,
    check_solution,
    build_plot
)


def build_c_system(x_values, y_values):
    validate_values(x_values, y_values)

    h_values = calculate_h_values(x_values)

    # Система имеет трёхдиагональный вид, поэтому матрица именно такая
    matrix = [
        [2 * (h_values[0] + h_values[1]), h_values[1], 0],
        [h_values[1], 2 * (h_values[1] + h_values[2]), h_values[2]],
        [0, h_values[2], 2 * (h_values[2] + h_values[3])]
    ]

    # средние наклоны функций на промежутках [xi-1, xi]
    # разность показывать характера изменения наклона между соседними промежутками
    right_side = [
        3 * ((y_values[2] - y_values[1]) / h_values[1] -
             (y_values[1] - y_values[0]) / h_values[0]),

        3 * ((y_values[3] - y_values[2]) / h_values[2] -
             (y_values[2] - y_values[1]) / h_values[1]),

        3 * ((y_values[4] - y_values[3]) / h_values[3] -
             (y_values[3] - y_values[2]) / h_values[2])
    ]

    return matrix, right_side


def calculate_c_values(x_values, y_values):
    matrix, right_side = build_c_system(x_values, y_values)

    # Решение системы A * c = b
    solution = solve_linear_system(matrix, right_side)

    # Сплайн имеет нулевую кривизну на левом конце, поэтому c1 = 0
    c_values = [0, solution[0], solution[1], solution[2]]

    return c_values, matrix, right_side


def calculate_a_values(y_values):
    # ai = f(xi-1)
    return [y_values[i] for i in range(len(y_values) - 1)]


def calculate_b_values(x_values, y_values, c_values):
    # bi - начальный наклон i-го сплайна
    h_values = calculate_h_values(x_values)
    n = len(x_values) - 1
    b_values = []

    for i in range(n - 1):
        # наклон между двумя табличными точками с учётом построения кубической гладкой кривой
        b = ((y_values[i + 1] - y_values[i]) / h_values[i] -
             h_values[i] * (c_values[i + 1] + 2 * c_values[i]) / 3)
        b_values.append(b)

    # т.к. S''(x4) = 0, то формула упрощается и считается по-другому
    b_last = ((y_values[n] - y_values[n - 1]) / h_values[n - 1] -
              2 * h_values[n - 1] * c_values[n - 1] / 3)
    b_values.append(b_last)

    return b_values


def calculate_d_values(x_values, c_values):
    # di - коэффициент при кубической степени i-го сплайна, управляет изменением кривизны
    h_values = calculate_h_values(x_values)
    n = len(x_values) - 1
    d_values = []

    for i in range(n - 1):
        d = (c_values[i + 1] - c_values[i]) / (3 * h_values[i])
        d_values.append(d)

    # т.к. S''(x4) = 0, то формула упрощается и считается по-другому
    d_last = -c_values[n - 1] / (3 * h_values[n - 1])
    d_values.append(d_last)

    return d_values


def calculate_spline_coefficients(x_values, y_values):
    c_values, matrix, right_side = calculate_c_values(x_values, y_values)
    a_values = calculate_a_values(y_values)
    b_values = calculate_b_values(x_values, y_values, c_values)
    d_values = calculate_d_values(x_values, c_values)

    return a_values, b_values, c_values, d_values, matrix, right_side


def spline_value(x, x_values, a_values, b_values, c_values, d_values):
    segment = find_segment(x, x_values)
    dx = x - x_values[segment]

    return (a_values[segment] +
            b_values[segment] * dx +
            c_values[segment] * dx ** 2 +
            d_values[segment] * dx ** 3)


def solve_case(x_values, y_values, x_star):
    print("Таблица значений:")
    print_values_table(x_values, y_values)
    print()

    h_values = calculate_h_values(x_values)
    print("Шаги сетки:")
    print_h_values(h_values)
    print()

    a_values, b_values, c_values, d_values, matrix, right_side = \
        calculate_spline_coefficients(x_values, y_values)

    print("Система для коэффициентов c2, c3, c4:")
    print_c_system(matrix, right_side)
    print()

    print("Коэффициенты кубического сплайна:")
    print_coefficients_table(x_values, a_values, b_values, c_values, d_values)
    print()

    print("Кубические сплайны:")
    print_spline_polynomials(x_values, a_values, b_values, c_values, d_values)
    print()

    result = spline_value(x_star, x_values, a_values,
                          b_values, c_values, d_values)

    print("Вычисление значения в точке:")
    print(f"X* = {x_star:.10f}")

    segment = find_segment(x_star, x_values)
    print(
        f"X* принадлежит отрезку [{x_values[segment]:.10f}, {x_values[segment + 1]:.10f}]")
    print(f"S{segment + 1}({x_star:.10f}) = {result:.10f}")
    print()

    print("Проверка:")
    check_solution(x_values, y_values, a_values, b_values, c_values, d_values)
    build_plot(x_values, y_values, a_values, b_values,
               c_values, d_values, x_star)
    print("-" * 50)


def main():
    x_star = 0.8

    x_values = [0.1, 0.5, 0.9, 1.3, 1.7]
    y_values = [10.0, 2.0, 1.1111, 0.76923, 0.58824]

    solve_case(x_values, y_values, x_star)


if __name__ == "__main__":
    main()
