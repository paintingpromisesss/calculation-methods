EPS = 1e-12


def validate_values(x_values, y_values, x_star):
    if len(x_values) != len(y_values):
        raise ValueError("Количество X и Y должно быть одинаковым")

    if len(x_values) != 5:
        raise ValueError("Для данной задачи должно быть задано 5 точек")

    for i in range(1, len(x_values)):
        if abs(x_values[i] - x_values[i - 1]) < EPS:
            raise ValueError("Узлы должны быть различными")

    point_index = find_point_index(x_values, x_star)

    if point_index == 0 or point_index == len(x_values) - 1:
        raise ValueError("Нужны соседние точки")


def find_point_index(x_values, x_star):
    for i in range(len(x_values)):
        if abs(x_values[i] - x_star) < EPS:
            return i

    raise ValueError("Точка X* должна совпадать с одним из узлов таблицы")


def calculate_left_derivative(x_values, y_values, point_index):
    # Формула (3.18) на отрезке [x(i-1), x(i)]
    return (
        (y_values[point_index] - y_values[point_index - 1]) /
        (x_values[point_index] - x_values[point_index - 1])
    )


def calculate_right_derivative(x_values, y_values, point_index):
    # Формула (3.18) на отрезке [x(i), x(i+1)]
    return (
        (y_values[point_index + 1] - y_values[point_index]) /
        (x_values[point_index + 1] - x_values[point_index])
    )


def calculate_first_derivative(x_values, y_values, point_index, x_star):
    # Формула (3.20), полученная дифференцированием
    # интерполяционного многочлена 2-ой степени
    x_prev = x_values[point_index - 1]
    x_curr = x_values[point_index]
    x_next = x_values[point_index + 1]

    left_derivative = calculate_left_derivative(
        x_values, y_values, point_index)

    right_derivative = calculate_right_derivative(
        x_values, y_values, point_index)

    return (
        left_derivative +
        (right_derivative - left_derivative) *
        (2 * x_star - x_prev - x_curr) /
        (x_next - x_prev)
    )


def calculate_second_derivative(x_values, y_values, point_index):
    # Формула (3.21), полученная вторым дифференцированием
    # интерполяционного многочлена 2-ой степени
    x_prev = x_values[point_index - 1]
    x_next = x_values[point_index + 1]

    left_derivative = calculate_left_derivative(
        x_values, y_values, point_index)

    right_derivative = calculate_right_derivative(
        x_values, y_values, point_index)

    return 2 * (right_derivative - left_derivative) / (x_next - x_prev)


def print_values_table(x_values, y_values):
    print("i\tXi\t\tYi")

    for i in range(len(x_values)):
        print(f"{i}\t{x_values[i]:.10f}\t{y_values[i]:.10f}")


def print_selected_points(x_values, y_values, point_index):
    print("i\tXi\t\tYi")

    for i in range(point_index - 1, point_index + 2):
        print(f"{i}\t{x_values[i]:.10f}\t{y_values[i]:.10f}")


def print_derivative_parts(left_derivative, right_derivative,
                           first_derivative, second_derivative):
    print(f"y'лев  = {left_derivative:.10f}")
    print(f"y'прав = {right_derivative:.10f}")
    print(f"y'     = {first_derivative:.10f}")
    print(f"y''    = {second_derivative:.10f}")


def print_result(x_star, first_derivative, second_derivative):
    print(f"y'({x_star:.10f})  = {first_derivative:.10f}")
    print(f"y''({x_star:.10f}) = {second_derivative:.10f}")

    