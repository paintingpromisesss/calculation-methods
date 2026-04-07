from utils import get_matrix_norm_c, get_vector_norm_c, print_vector, check_solution, fix_zero_diagonal, prepare_system

def error_estimate(alpha_norm, x_new, x_old):
    diff = [x_new[i] - x_old[i] for i in range(len(x_new))]
    diff_norm = get_vector_norm_c(diff)

    if alpha_norm < 1:
        return alpha_norm / (1 - alpha_norm) * diff_norm
    return diff_norm


def simple_iteration_method(a, b, eps, max_iterations=10000):
    n = len(b)
    alpha, beta = prepare_system(a, b, eps, n)

    alpha_norm = get_matrix_norm_c(alpha)
    if alpha_norm >= 1:
        print("Метод простых итераций не может гарантировать сходимость (||alpha|| >= 1)")

    x_old = beta[:]
    for iteration in range(1, max_iterations+1):
        x_new = [0.0] * n
        for i in range(n):
            x_new[i] = beta[i] + sum(alpha[i][j] * x_old[j] for j in range(n))
        err = error_estimate(alpha_norm, x_new, x_old)
        if err < eps:
            return x_new, iteration, err, alpha_norm
        x_old = x_new

    raise ValueError("Метод простых итераций не сошелся за максимальное количество итераций")


def main():
    a = [
        [-24.0, -6.0, 4.0, 7.0],
        [-8.0, 21.0, 4.0, -2.0],
        [6.0, 6.0, 16.0, 0.0],
        [-7.0, -7.0, 5.0, 24.0],
    ]
    b = [130.0, 139.0, -84.0, -165.0]

    zero_eps = 1e-12
    iter_eps = 1e-6

    a, b = fix_zero_diagonal(a, b, zero_eps)
    x, iterations, err, alpha_norm = simple_iteration_method(a, b, iter_eps)

    print(f"||alpha|| = {alpha_norm:.10f}")
    print(f"Число итераций: {iterations}")
    print(f"Оценка погрешности: {err:.16e}\n")

    print_vector("Решение:", x, "x")
    check_solution(a, b, x)

if __name__ == "__main__":
    main()
