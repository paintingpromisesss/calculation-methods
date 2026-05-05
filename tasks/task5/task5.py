from utils import (
    validate_values,
    generate_x_values,
    generate_midpoints,
    calculate_y_values,
    calculate_rectangles_integral,
    calculate_trapezoids_integral,
    calculate_simpson_integral,
    calculate_runge_romberg_value,
    calculate_runge_romberg_error,
    print_values_table,
    print_midpoints_table,
    print_integral_parts,
    print_runge_romberg_parts,
    print_result
)

from math import sqrt


def function(x):
    return 1 / sqrt((2 * x + 7) * (3 * x + 4))


def calculate_integrals(function, x_start, x_end, step):
    validate_values(x_start, x_end, step)

    x_values = generate_x_values(x_start, x_end, step)
    y_values = calculate_y_values(function, x_values)

    midpoints = generate_midpoints(x_values)
    midpoint_y_values = calculate_y_values(function, midpoints)

    # Метод прямоугольников по формуле (3.23)
    rectangles_integral = calculate_rectangles_integral(
        midpoint_y_values, step)

    # Метод трапеций по формуле (3.25)
    trapezoids_integral = calculate_trapezoids_integral(
        y_values, step)

    # Метод Симпсона по формуле (3.28)
    simpson_integral = calculate_simpson_integral(
        y_values, step)

    return {
        "x_values": x_values,
        "y_values": y_values,
        "midpoints": midpoints,
        "midpoint_y_values": midpoint_y_values,
        "rectangles_integral": rectangles_integral,
        "trapezoids_integral": trapezoids_integral,
        "simpson_integral": simpson_integral
    }


def calculate_runge_romberg(integrals_h1, integrals_h2, h1, h2):
    k = h1 / h2

    # Для метода прямоугольников порядок точности p = 2
    rectangles_value = calculate_runge_romberg_value(
        integrals_h2["rectangles_integral"],
        integrals_h1["rectangles_integral"],
        k,
        2
    )

    rectangles_error = calculate_runge_romberg_error(
        integrals_h2["rectangles_integral"],
        integrals_h1["rectangles_integral"],
        k,
        2
    )

    # Для метода трапеций порядок точности p = 2
    trapezoids_value = calculate_runge_romberg_value(
        integrals_h2["trapezoids_integral"],
        integrals_h1["trapezoids_integral"],
        k,
        2
    )

    trapezoids_error = calculate_runge_romberg_error(
        integrals_h2["trapezoids_integral"],
        integrals_h1["trapezoids_integral"],
        k,
        2
    )

    # Для метода Симпсона порядок точности p = 4
    simpson_value = calculate_runge_romberg_value(
        integrals_h2["simpson_integral"],
        integrals_h1["simpson_integral"],
        k,
        4
    )

    simpson_error = calculate_runge_romberg_error(
        integrals_h2["simpson_integral"],
        integrals_h1["simpson_integral"],
        k,
        4
    )

    return {
        "k": k,
        "rectangles_value": rectangles_value,
        "rectangles_error": rectangles_error,
        "trapezoids_value": trapezoids_value,
        "trapezoids_error": trapezoids_error,
        "simpson_value": simpson_value,
        "simpson_error": simpson_error
    }


def solve_case(function, x_start, x_end, h1, h2):
    print("y = 1 / sqrt((2x + 7)(3x + 4))")
    print(f"X0 = {x_start:.10f}")
    print(f"Xk = {x_end:.10f}")
    print(f"h1 = {h1:.10f}")
    print(f"h2 = {h2:.10f}")
    print()

    print(f"Расчет с шагом h1 = {h1:.10f}")
    integrals_h1 = calculate_integrals(function, x_start, x_end, h1)

    print("Таблица значений:")
    print_values_table(integrals_h1["x_values"], integrals_h1["y_values"])
    print()

    print("Таблица середин отрезков для метода прямоугольников:")
    print_midpoints_table(
        integrals_h1["midpoints"],
        integrals_h1["midpoint_y_values"]
    )
    print()

    print("Интегралы:")
    print_integral_parts(
        integrals_h1["rectangles_integral"],
        integrals_h1["trapezoids_integral"],
        integrals_h1["simpson_integral"]
    )
    print()

    print(f"Расчет с шагом h2 = {h2:.10f}")
    integrals_h2 = calculate_integrals(function, x_start, x_end, h2)

    print("Таблица значений:")
    print_values_table(integrals_h2["x_values"], integrals_h2["y_values"])
    print()

    print("Таблица середин отрезков для метода прямоугольников:")
    print_midpoints_table(
        integrals_h2["midpoints"],
        integrals_h2["midpoint_y_values"]
    )
    print()

    print("Интегралы:")
    print_integral_parts(
        integrals_h2["rectangles_integral"],
        integrals_h2["trapezoids_integral"],
        integrals_h2["simpson_integral"]
    )
    print()

    runge_romberg = calculate_runge_romberg(
        integrals_h1, integrals_h2, h1, h2)

    print("Уточнение по методу Рунге-Ромберга-Ричардсона:")
    print_runge_romberg_parts(runge_romberg)
    print()

    print("Ответ:")
    print_result(integrals_h1, integrals_h2, runge_romberg)


def main():
    x_start = 0.0
    x_end = 4.0
    h1 = 1.0
    h2 = 0.5

    solve_case(function, x_start, x_end, h1, h2)


if __name__ == "__main__":
    main()