EPS = 1e-12


def validate_values(x_start, x_end, step):
    if step <= 0:
        raise ValueError("Шаг должен быть положительным")

    if x_end <= x_start:
        raise ValueError("Правая граница должна быть больше левой")

    intervals_count = (x_end - x_start) / step

    if abs(intervals_count - round(intervals_count)) > EPS:
        raise ValueError("Отрезок должен делиться на целое число шагов")

    if int(round(intervals_count)) % 2 != 0:
        raise ValueError(
            "Для метода Симпсона число интервалов должно быть четным")


def generate_x_values(x_start, x_end, step):
    intervals_count = int(round((x_end - x_start) / step))

    return [
        x_start + i * step
        for i in range(intervals_count + 1)
    ]


def generate_midpoints(x_values):
    return [
        (x_values[i - 1] + x_values[i]) / 2
        for i in range(1, len(x_values))
    ]


def calculate_y_values(function, x_values):
    return [
        function(x)
        for x in x_values
    ]


def calculate_rectangles_integral(midpoint_y_values, step):
    # Формула (3.23):
    # F = h * sum(f((x(i-1) + x(i)) / 2))
    return step * sum(midpoint_y_values)


def calculate_trapezoids_integral(y_values, step):
    # Формула (3.25):
    # F = h * (y0 / 2 + y1 + ... + y(n-1) + yn / 2)
    return step * (
        y_values[0] / 2 +
        sum(y_values[1:-1]) +
        y_values[-1] / 2
    )


# приближает график функции кусками парабол. На каждые два соседних интервала берутся 3 точки.
# Через них проводится парабола, интеграл от паработлы на участке даёт веса 1 4 1, который для нескольких последовательных пар дают коэффы 2 на нечётных индексах

def calculate_simpson_integral(y_values, step):
    # Формула (3.28):
    # F = h / 3 * (y0 + 4y1 + 2y2 + ... + 4y(n-1) + yn)
    # у крайних точек коэффициент 1
    result = y_values[0] + y_values[-1]

    for i in range(1, len(y_values) - 1):
        if i % 2 == 1:
            result += 4 * y_values[i]
        else:
            result += 2 * y_values[i]

    return step * result / 3


# Метод Рунге-Ромберга уточняет результат,
# используя два значения интеграла: с большим и меньшим шагом.
# Если порядок точности метода равен p, то главная ошибка зависит от h^p.
# Поэтому по разности F(h) и F(kh) можно оценить эту ошибку
# и добавить поправку к результату с меньшим шагом.
def calculate_runge_romberg_value(f_h, f_kh, k, p):
    # Формула (3.30):
    # F = Fh + (Fh - Fkh) / (k^p - 1)
    return f_h + (f_h - f_kh) / (k ** p - 1)


def calculate_runge_romberg_error(f_h, f_kh, k, p):
    return abs((f_h - f_kh) / (k ** p - 1))


def print_values_table(x_values, y_values):
    print("i\tXi\t\tYi")

    for i in range(len(x_values)):
        print(f"{i}\t{x_values[i]:.10f}\t{y_values[i]:.10f}")


def print_midpoints_table(midpoints, midpoint_y_values):
    print("i\tXср\t\tYср")

    for i in range(len(midpoints)):
        print(f"{i + 1}\t{midpoints[i]:.10f}\t{midpoint_y_values[i]:.10f}")


def print_integral_parts(rectangles_integral,
                         trapezoids_integral,
                         simpson_integral):
    print(f"Метод прямоугольников = {rectangles_integral:.10f}")
    print(f"Метод трапеций        = {trapezoids_integral:.10f}")
    print(f"Метод Симпсона        = {simpson_integral:.10f}")


def print_runge_romberg_parts(runge_romberg):
    print(f"k = {runge_romberg['k']:.10f}")
    print()
    print("Метод прямоугольников:")
    print(f"F = {runge_romberg['rectangles_value']:.10f}")
    print(f"R = {runge_romberg['rectangles_error']:.10f}")
    print()
    print("Метод трапеций:")
    print(f"F = {runge_romberg['trapezoids_value']:.10f}")
    print(f"R = {runge_romberg['trapezoids_error']:.10f}")
    print()
    print("Метод Симпсона:")
    print(f"F = {runge_romberg['simpson_value']:.10f}")
    print(f"R = {runge_romberg['simpson_error']:.10f}")


def print_result(integrals_h1, integrals_h2, runge_romberg):
    print("Метод\t\t\tF(h1)\t\tF(h2)\t\tF уточн.\tR")

    print(
        "Прямоугольников\t\t"
        f"{integrals_h1['rectangles_integral']:.10f}\t"
        f"{integrals_h2['rectangles_integral']:.10f}\t"
        f"{runge_romberg['rectangles_value']:.10f}\t"
        f"{runge_romberg['rectangles_error']:.10f}"
    )

    print(
        "Трапеций\t\t"
        f"{integrals_h1['trapezoids_integral']:.10f}\t"
        f"{integrals_h2['trapezoids_integral']:.10f}\t"
        f"{runge_romberg['trapezoids_value']:.10f}\t"
        f"{runge_romberg['trapezoids_error']:.10f}"
    )

    print(
        "Симпсона\t\t"
        f"{integrals_h1['simpson_integral']:.10f}\t"
        f"{integrals_h2['simpson_integral']:.10f}\t"
        f"{runge_romberg['simpson_value']:.10f}\t"
        f"{runge_romberg['simpson_error']:.10f}"
    )
