import os
import matplotlib.pyplot as plt

EPS = 1e-12


def get_plot_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def validate_values(x_values, y_values):
    if len(x_values) != len(y_values):
        raise ValueError("Количество X и Y должно быть одинаковым")

    if len(x_values) != 5:
        raise ValueError("Для данной задачи должно быть задано 5 узлов")

    for i in range(1, len(x_values)):
        if x_values[i] <= x_values[i - 1]:
            raise ValueError(
                "Узлы интерполяции должны быть расположены по возрастанию")


def calculate_h_values(x_values):
    h_values = []

    for i in range(1, len(x_values)):
        h = x_values[i] - x_values[i - 1]

        if abs(h) < EPS:
            raise ValueError("Узлы интерполяции должны быть различными")

        h_values.append(h)

    return h_values


def solve_linear_system(matrix, right_side, eps=EPS):
    n = len(right_side)

    a = [row[:] for row in matrix]
    b = right_side[:]

    # Прямой ход метода Гаусса
    for i in range(n):
        max_row = i

        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[max_row][i]):
                max_row = j

        if abs(a[max_row][i]) < eps:
            raise ValueError("Система не имеет единственного решения")

        a[i], a[max_row] = a[max_row], a[i]
        b[i], b[max_row] = b[max_row], b[i]

        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]

            for k in range(i, n):
                a[j][k] -= factor * a[i][k]

            b[j] -= factor * b[i]

    # Обратный ход метода Гаусса
    solution = [0 for _ in range(n)]

    for i in range(n - 1, -1, -1):
        value = b[i]

        for j in range(i + 1, n):
            value -= a[i][j] * solution[j]

        if abs(a[i][i]) < eps:
            raise ValueError("Система не имеет единственного решения")

        solution[i] = value / a[i][i]

    return solution

def find_segment(x, x_values):
    if x < x_values[0] - EPS or x > x_values[-1] + EPS:
        raise ValueError("Точка X* находится вне отрезка интерполяции")

    for i in range(len(x_values) - 1):
        if x_values[i] - EPS <= x <= x_values[i + 1] + EPS:
            return i

    return len(x_values) - 2


def print_values_table(x_values, y_values):
    print("i\tXi\t\tYi")

    for i in range(len(x_values)):
        print(f"{i}\t{x_values[i]:.10f}\t{y_values[i]:.10f}")


def print_h_values(h_values):
    for i in range(len(h_values)):
        print(f"h{i + 1} = {h_values[i]:.10f}")


def print_c_system(matrix, right_side):
    names = ["c2", "c3", "c4"]

    for i in range(len(matrix)):
        row = ""

        for j in range(len(matrix[i])):
            coefficient = matrix[i][j]

            if abs(coefficient) < EPS:
                continue

            if row:
                row += " + "

            row += f"{coefficient:.10f}*{names[j]}"

        row += f" = {right_side[i]:.10f}"

        print(row)


def print_coefficients_table(x_values, a_values, b_values, c_values, d_values):
    print("i\t[xi-1,           xi]\t\tai\t\tbi\t\tci\t\tdi")

    for i in range(len(a_values)):
        print(f"{i + 1}\t"
              f"[{x_values[i]:.10f}, {x_values[i + 1]:.10f}]\t"
              f"{a_values[i]:.10f}\t"
              f"{b_values[i]:.10f}\t"
              f"{c_values[i]:.10f}\t"
              f"{d_values[i]:.10f}")


def print_spline_polynomials(x_values, a_values, b_values, c_values, d_values):
    for i in range(len(a_values)):
        print(f"S{i + 1}(x) = "
              f"{a_values[i]:.10f} + "
              f"({b_values[i]:.10f})*(x - {x_values[i]:.10f}) + "
              f"({c_values[i]:.10f})*(x - {x_values[i]:.10f})^2 + "
              f"({d_values[i]:.10f})*(x - {x_values[i]:.10f})^3, "
              f"{x_values[i]:.10f} <= x <= {x_values[i + 1]:.10f}")


def check_solution(x_values, y_values, a_values, b_values, c_values, d_values):
    for i in range(len(a_values)):
        left_x = x_values[i]
        right_x = x_values[i + 1]

        left_value = (a_values[i] +
                      b_values[i] * 0 +
                      c_values[i] * 0 ** 2 +
                      d_values[i] * 0 ** 3)

        dx = right_x - left_x
        right_value = (a_values[i] +
                       b_values[i] * dx +
                       c_values[i] * dx ** 2 +
                       d_values[i] * dx ** 3)

        print(f"|S{i + 1}({left_x:.10f}) - f{i}| = "
              f"{abs(left_value - y_values[i]):.10f}")

        print(f"|S{i + 1}({right_x:.10f}) - f{i + 1}| = "
              f"{abs(right_value - y_values[i + 1]):.10f}")


def calculate_spline_value(x, x_values, a_values, b_values, c_values, d_values):
    segment = find_segment(x, x_values)
    dx = x - x_values[segment]

    return (a_values[segment] +
            b_values[segment] * dx +
            c_values[segment] * dx ** 2 +
            d_values[segment] * dx ** 3)


def build_plot(x_values, y_values, a_values, b_values, c_values, d_values, x_star):
    left = min(x_values)
    right = max(x_values)
    points_count = 200
    step = (right - left) / (points_count - 1)

    x_plot = []
    y_spline = []

    for i in range(points_count):
        x = left + step * i

        x_plot.append(x)
        y_spline.append(calculate_spline_value(
            x, x_values, a_values, b_values, c_values, d_values))

    y_star = calculate_spline_value(
        x_star, x_values, a_values, b_values, c_values, d_values)

    file_name = get_plot_path("task2_plot.png")

    plt.figure()
    plt.scatter(x_values, y_values, label="Табличные значения", zorder=3)
    plt.scatter([x_star], [y_star], label="X*", zorder=4)
    plt.plot(x_plot, y_spline, label="S(x)")

    plt.title("Задание 3.2")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()
