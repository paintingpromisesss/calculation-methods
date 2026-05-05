from utils import (
    validate_values,
    find_point_index,
    calculate_left_derivative,
    calculate_right_derivative,
    calculate_first_derivative,
    calculate_second_derivative,
    print_values_table,
    print_selected_points,
    print_derivative_parts,
    print_result
)


def calculate_derivatives(x_values, y_values, x_star):
    validate_values(x_values, y_values, x_star)

    point_index = find_point_index(x_values, x_star)

    # Производная слева по формуле (3.18)
    left_derivative = calculate_left_derivative(
        x_values, y_values, point_index)

    # Производная справа по формуле (3.18)
    right_derivative = calculate_right_derivative(
        x_values, y_values, point_index)

    # Первая производная по формуле (3.20)
    first_derivative = calculate_first_derivative(
        x_values, y_values, point_index, x_star)

    # Вторая производная по формуле (3.21)
    second_derivative = calculate_second_derivative(
        x_values, y_values, point_index)

    return {
        "point_index": point_index,
        "left_derivative": left_derivative,
        "right_derivative": right_derivative,
        "first_derivative": first_derivative,
        "second_derivative": second_derivative
    }


def solve_case(x_values, y_values, x_star):
    print("Таблица значений:")
    print_values_table(x_values, y_values)
    print()

    result = calculate_derivatives(x_values, y_values, x_star)
    point_index = result["point_index"]

    print(f"Точка вычисления: X* = {x_star:.10f}")
    print(f"X* совпадает с узлом x{point_index}")
    print()

    print("Три точки для интерполяционного многочлена 2-ой степени:")
    print_selected_points(x_values, y_values, point_index)
    print()

    print("Производные по формулам методички:")
    print_derivative_parts(
        result["left_derivative"],
        result["right_derivative"],
        result["first_derivative"],
        result["second_derivative"]
    )
    print()

    print("Ответ:")
    print_result(x_star, result["first_derivative"], result["second_derivative"])


def main():
    x_star = 2.0

    x_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    y_values = [2.0, 2.1667, 2.5, 2.9, 3.3333]

    solve_case(x_values, y_values, x_star)


if __name__ == "__main__":
    main()