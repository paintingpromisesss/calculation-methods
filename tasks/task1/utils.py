EPS = 1e-12


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