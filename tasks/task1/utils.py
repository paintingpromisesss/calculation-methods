import os
import matplotlib.pyplot as plt

EPS = 1e-12


def get_plot_path(file_name):
    return os.path.join(os.path.dirname(__file__), file_name)


def f(x):
    if abs(x) < EPS:
        raise ValueError("x должен быть отличен от нуля")
    return 1 / x

def calculate_y_values(x_values):
    return [f(x) for x in x_values]

def lagrange_basis(x, x_values, i):
    result = 1

    for j in range(len(x_values)):
        if i != j:
            denominator = x_values[i] - x_values[j]

            if abs(denominator) < EPS:
                raise ValueError("Узлы интерполяции должны быть различными")
            
            result *= (x - x_values[j]) / denominator
    
    return result

def absolute_error(exact_value, approx_value):
    return abs(exact_value - approx_value)

def print_values_table(x_values, y_values):
    print("i\tXi\t\tYi")

    for i in range(len(x_values)):
        print(f"{i}\t{x_values[i]:.10f}\t{y_values[i]:.10f}")


def print_divided_differences_table(x_values, table):
    print("i\tXi\t\tf(Xi)\t\t1 порядок\t2 порядок\t3 порядок")

    n = len(x_values)

    for i in range(n):
        row = f"{i}\t{x_values[i]:.10f}"

        for j in range(n - i):
            row += f"\t{table[i][j]:.10f}"

        print(row)

def print_polynomial_coefficients(coefficients):
    for i in range(len(coefficients)):
        print(f"a{i} = {coefficients[i]:.10f}")

def check_solution(x_star, lagrange_value, newton_value):
    print(f"|L3({x_star:.10f}) - P3({x_star:.10f})| = "
          f"{abs(lagrange_value - newton_value):.10f}")


def calculate_lagrange_value(x, x_values, y_values):
    result = 0

    for i in range(len(x_values)):
        result += y_values[i] * lagrange_basis(x, x_values, i)

    return result


def calculate_newton_value(x, x_values, divided_difference_table):
    result = divided_difference_table[0][0]
    product = 1

    for i in range(1, len(x_values)):
        product *= (x - x_values[i - 1])
        result += divided_difference_table[0][i] * product

    return result


def build_plot(case_name, x_values, y_values, divided_difference_table, x_star):
    left = min(x_values)
    right = max(x_values)
    points_count = 200
    step = (right - left) / (points_count - 1)

    x_plot = []
    y_function = []
    y_lagrange = []
    y_newton = []

    for i in range(points_count):
        x = left + step * i

        x_plot.append(x)
        y_function.append(f(x))
        y_lagrange.append(calculate_lagrange_value(x, x_values, y_values))
        y_newton.append(calculate_newton_value(
            x, x_values, divided_difference_table))

    file_name = get_plot_path(f"task1_plot_{case_name}.png")

    plt.figure()
    plt.scatter(x_values, y_values, label="Узлы интерполяции", zorder=3)
    plt.scatter([x_star], [f(x_star)], label="X*", zorder=4)
    plt.plot(x_plot, y_function, label="f(x)")
    plt.plot(x_plot, y_lagrange, label="L3(x)")
    plt.plot(x_plot, y_newton, "--", label="P3(x)")

    plt.title(f"Задание 3.1, случай {case_name}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    plt.savefig(file_name, dpi=300, bbox_inches="tight")
    plt.close()
    
