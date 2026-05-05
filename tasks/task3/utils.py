import matplotlib.pyplot as plt
EPS = 1e-12

def validate_values(x_values, y_values):
    if len(x_values) != len(y_values):
        raise ValueError("Количество X и Y должно быть одинаковым")

    if len(x_values) != 6:
        raise ValueError("Для данной задачи должно быть задано 6 точек")

    for i in range(1, len(x_values)):
        if abs(x_values[i] - x_values[i - 1]) < EPS:
            raise ValueError("Узлы должны быть различными")


def calculate_sums(x_values, y_values):
    sums = {
        "n": len(x_values),
        "sum_x": 0,
        "sum_x2": 0,
        "sum_x3": 0,
        "sum_x4": 0,
        "sum_y": 0,
        "sum_xy": 0,
        "sum_x2y": 0
    }

    for i in range(len(x_values)):
        x = x_values[i]
        y = y_values[i]

        sums["sum_x"] += x
        sums["sum_x2"] += x ** 2
        sums["sum_x3"] += x ** 3
        sums["sum_x4"] += x ** 4
        sums["sum_y"] += y
        sums["sum_xy"] += x * y
        sums["sum_x2y"] += x ** 2 * y
    
    return sums

def build_linear_system(sums):
    # F1(x) = a0 + a1*x
    # a0*N + a1*sum(xi) = sum(yi)
    # a0*sum(xi) + a1*sum(xi^2) = sum(xi*yi)
    matrix = [
        [sums["n"], sums["sum_x"]],
        [sums["sum_x"], sums["sum_x2"]]
    ]

    right_side = [
        sums["sum_y"],
        sums["sum_xy"]
    ]

    return matrix, right_side

def build_quadratic_system(sums):
    # F2(x) = a0 + a1*x + a2*x^2
    # a0*N + a1*sum(xi) + a2*sum(xi^2) = sum(yi)
    # a0*sum(xi) + a1*sum(xi^2) + a2*sum(xi^3) = sum(xi*yi)
    # a0*sum(xi^2) + a1*sum(xi^3) + a2*sum(xi^4) = sum(xi^2*yi)
    matrix = [
        [sums["n"], sums["sum_x"], sums["sum_x2"]],
        [sums["sum_x"], sums["sum_x2"], sums["sum_x3"]],
        [sums["sum_x2"], sums["sum_x3"], sums["sum_x4"]]
    ]

    right_side = [
        sums["sum_y"],
        sums["sum_xy"],
        sums["sum_x2y"]
    ]

    return matrix, right_side

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


def polynomial_value(x, coefficients):
    value = 0

    for i in range(len(coefficients)):
        value += coefficients[i] * x ** i

    return value

def calculate_error_sum(x_values, y_values, coefficients):
    error_sum = 0

    for i in range(len(x_values)):
        approximation = polynomial_value(x_values[i], coefficients)
        error = approximation - y_values[i]
        error_sum += error ** 2

    return error_sum


def print_values_table(x_values, y_values):
    print("i\tXi\t\tYi")

    for i in range(len(x_values)):
        print(f"{i}\t{x_values[i]:.10f}\t{y_values[i]:.10f}")


def print_sums(sums):
    print(f"Σ(xi)      = {sums['sum_x']:.10f}")
    print(f"Σ(xi^2)    = {sums['sum_x2']:.10f}")
    print(f"Σ(xi^3)    = {sums['sum_x3']:.10f}")
    print(f"Σ(xi^4)    = {sums['sum_x4']:.10f}")
    print(f"Σ(yi)      = {sums['sum_y']:.10f}")
    print(f"Σ(xi*yi)   = {sums['sum_xy']:.10f}")
    print(f"Σ(xi^2*yi) = {sums['sum_x2y']:.10f}")


def print_system(matrix, right_side, names):
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


def print_polynomial(coefficients, name):
    polynomial = f"{name}(x) = "

    for i in range(len(coefficients)):
        coefficient = coefficients[i]

        if i == 0:
            polynomial += f"{coefficient:.10f}"
        else:
            if coefficient >= 0:
                polynomial += " + "
            else:
                polynomial += " - "

            polynomial += f"{abs(coefficient):.10f}*x"

            if i > 1:
                polynomial += f"^{i}"

    print(polynomial)


def print_approximation_table(x_values, y_values, coefficients):
    print("i\tXi\t\tYi\t\tF(Xi)\t\tF(Xi)-Yi\t(F(Xi)-Yi)^2")

    for i in range(len(x_values)):
        approximation = polynomial_value(x_values[i], coefficients)
        error = approximation - y_values[i]

        print(f"{i}\t"
              f"{x_values[i]:.10f}\t"
              f"{y_values[i]:.10f}\t"
              f"{approximation:.10f}\t"
              f"{error:.10f}\t"
              f"{error ** 2:.10f}")


def print_result_comparison(linear_error_sum, quadratic_error_sum):
    print(f"Ф1 = {linear_error_sum:.10f}")
    print(f"Ф2 = {quadratic_error_sum:.10f}")

    if linear_error_sum < quadratic_error_sum:
        print("Многочлен 1-ой степени даёт меньшую сумму квадратов ошибок")
    elif quadratic_error_sum < linear_error_sum:
        print("Многочлен 2-ой степени даёт меньшую сумму квадратов ошибок")
    else:
        print("Суммы квадратов ошибок равны")


def build_plot(x_values, y_values, linear_coefficients, quadratic_coefficients):

    left = min(x_values)
    right = max(x_values)
    points_count = 200
    step = (right - left) / (points_count - 1)

    x_plot = []
    y_linear = []
    y_quadratic = []

    for i in range(points_count):
        x = left + step * i

        x_plot.append(x)
        y_linear.append(polynomial_value(x, linear_coefficients))
        y_quadratic.append(polynomial_value(x, quadratic_coefficients))

    plt.scatter(x_values, y_values, label="Табличные значения")
    plt.plot(x_plot, y_linear, label="F1(x)")
    plt.plot(x_plot, y_quadratic, label="F2(x)")

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)
    plt.legend()
    # plt.savefig("task3_plot.png", dpi=300)