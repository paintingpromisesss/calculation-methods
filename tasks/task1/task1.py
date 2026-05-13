from utils import (
    EPS,
    lagrange_basis,
    calculate_y_values,
    print_values_table,
    f,
    absolute_error,
    print_divided_differences_table,
    print_polynomial_coefficients,
    check_solution,
    build_plot
)

def lagrange_polynomial(x, x_values, y_values):
    if len(x_values) != len(y_values):
        raise ValueError("Количество X и Y должно быть одинаковым")
    
    result = 0

    for i in range(len(x_values)):
        result += y_values[i] * lagrange_basis(x, x_values, i)

    return result

def divided_differences(x_values, y_values, eps=EPS):
    if len(x_values) != len(y_values):
        raise ValueError("Количество X и Y должно быть одинаковым")
    
    n = len(x_values)
    table = [[0 for _ in range(n)] for _ in range(n)]

    for i in range(n):
        table[i][0] = y_values[i]
    
    for j in range(1, n):
        for i in range(n-j):
            denominator = x_values[i+j] - x_values[i]

            if abs(denominator) < eps:
                raise ValueError("Узлы интерполяции должны быть различными")
            
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / denominator
    
    return table

def newton_polynomial(x, x_values, divided_difference_table):
    result = divided_difference_table[0][0]
    product = 1

    for i in range(1, len(x_values)):
        product *= (x - x_values[i-1])
        result += divided_difference_table[0][i] * product

    return result

def solve_case(case_name, x_values, x_star):
    print(f"Случай {case_name}:")
    print() 

    y_values = calculate_y_values(x_values)
    print("Таблица значений:")
    print_values_table(x_values, y_values)
    print()

    lagrange_value = lagrange_polynomial(x_star, x_values, y_values)
    exact_value = f(x_star)
    lagrange_error = absolute_error(exact_value, lagrange_value)

    print("Многочлен Лагранжа:")
    print(f"L3({x_star:.10f}) = {lagrange_value:.10f}")
    print(f"f({x_star:.10f}) = {exact_value:.10f}")
    print(f"Погрешность: {lagrange_error:.10f}")
    print()

    table = divided_differences(x_values, y_values)
    print("Таблица разделенных разностей:")
    print_divided_differences_table(x_values, table)
    print()

    coefficients = [table[0][i] for i in range(len(x_values))]
    print("Коэффициенты многочлена Ньютона:")
    print_polynomial_coefficients(coefficients)
    print()

    newton_value = newton_polynomial(x_star, x_values, table)
    newton_error = absolute_error(exact_value, newton_value)

    print("Многочлен Ньютона:")
    print(f"P3({x_star:.10f}) = {newton_value:.10f}")
    print(f"f({x_star:.10f}) = {exact_value:.10f}")
    print(f"Погрешность: {newton_error:.10f}")
    print()

    print("Проверка совпадения значений:")
    check_solution(x_star, lagrange_value, newton_value)
    build_plot(case_name, x_values, y_values, table, x_star)
    print("-" * 50)

def main():
    x_star = 0.8

    x_values_a = [0.1, 0.5, 0.9, 1.3]
    x_values_b = [0.1, 0.5, 1.1, 1.3]

    solve_case("a", x_values_a, x_star)
    solve_case("b", x_values_b, x_star)

if __name__ == "__main__":
    main()
