from utils import (
    validate_values,
    calculate_sums,
    solve_linear_system,
    calculate_error_sum,
    build_linear_system,
    build_quadratic_system,
    print_values_table,
    print_sums,
    print_system,
    print_polynomial,
    print_approximation_table,
    print_result_comparison,
    build_plot
)


def calculate_linear_polynomial(x_values, y_values):
    sums = calculate_sums(x_values, y_values)
    matrix, right_side = build_linear_system(sums)

    # Решение нормальной системы МНК для многочлена 1-ой степени
    coefficients = solve_linear_system(matrix, right_side)

    return coefficients, matrix, right_side, sums


def calculate_quadratic_polynomial(x_values, y_values):
    sums = calculate_sums(x_values, y_values)
    matrix, right_side = build_quadratic_system(sums)

    # Решение нормальной системы МНК для многочлена 2-ой степени
    coefficients = solve_linear_system(matrix, right_side)

    return coefficients, matrix, right_side, sums


def solve_case(x_values, y_values):
    validate_values(x_values, y_values)

    print("Таблица значений:")
    print_values_table(x_values, y_values)
    print()

    linear_coefficients, linear_matrix, linear_right_side, sums = \
        calculate_linear_polynomial(x_values, y_values)

    print("Суммы для нормальных систем МНК:")
    print_sums(sums)
    print()

    print("Нормальная система для многочлена 1-ой степени:")
    print_system(linear_matrix, linear_right_side, ["a0", "a1"])
    print()

    print("Приближающий многочлен 1-ой степени:")
    print_polynomial(linear_coefficients, "F1")
    print()

    linear_error_sum = calculate_error_sum(
        x_values, y_values, linear_coefficients)

    print("Таблица значений и ошибок для многочлена 1-ой степени:")
    print_approximation_table(x_values, y_values, linear_coefficients)
    print(f"Ф1 = {linear_error_sum:.10f}")
    print("-" * 50)

    quadratic_coefficients, quadratic_matrix, quadratic_right_side, _ = \
        calculate_quadratic_polynomial(x_values, y_values)

    print("Нормальная система для многочлена 2-ой степени:")
    print_system(quadratic_matrix, quadratic_right_side, ["a0", "a1", "a2"])
    print()

    print("Приближающий многочлен 2-ой степени:")
    print_polynomial(quadratic_coefficients, "F2")
    print()

    quadratic_error_sum = calculate_error_sum(
        x_values, y_values, quadratic_coefficients)

    print("Таблица значений и ошибок для многочлена 2-ой степени:")
    print_approximation_table(x_values, y_values, quadratic_coefficients)
    print(f"Ф2 = {quadratic_error_sum:.10f}")
    print("-" * 50)

    print("Сравнение результатов:")
    print_result_comparison(linear_error_sum, quadratic_error_sum)
    print()

    build_plot(x_values, y_values, linear_coefficients, quadratic_coefficients)


def main():
    x_values = [0.1, 0.5, 0.9, 1.3, 1.7, 2.1]
    y_values = [10.0, 2.0, 1.1111, 0.76923, 0.58824, 0.47619]

    solve_case(x_values, y_values)


if __name__ == "__main__":
    main()