from utils import EPS, dphi, phi, f, df, get_x0_newton, get_x0_simple_iteration

def simple_iteration(a, b, eps=EPS, max_iter=1000):
    x0 = get_x0_simple_iteration(a, b)
    if eps <= 0:
        raise ValueError("Эпсилон должен быть положительным")
    
    q = max(abs(dphi(x)) for x in [a, b])

    if q >= 1:
        raise ValueError(f"Метод простых итераций не гарантирует сходимость: q = {q:.6f} >= 1")

    x_prev = x0

    for k in range(1, max_iter + 1):
        x_next = phi(x_prev)
        diff = abs(x_next - x_prev)

        error_estimate = q / (1 - q) * diff
        if error_estimate <= eps:
            return x_next, q, k

        x_prev = x_next

    raise ValueError(
        "Метод простых итераций не сошелся за максимальное количество итераций")


def newton_method(a, b, eps=EPS, max_iter=1000, x0=None):
    if x0 is None:
        x0 = get_x0_newton(a, b)
    if eps <= 0:
        raise ValueError("Эпсилон должен быть положительным")

    x = x0
    for k in range(1, max_iter + 1):
        print(f"Итерация {k}: x = {x}")
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < eps:
            raise ValueError("Производная слишком мала, метод Ньютона может не сойтись")

        x_next = x - fx / dfx
        diff = abs(x_next - x)

        if diff < eps:
            return x_next, k

        x = x_next
    raise ValueError(
        "Метод Ньютона не сошелся за максимальное количество итераций")


def main():
    a, b = 0.0, 0.5
    root, q, iterations = simple_iteration(a, b, max_iter=1000)

    print("Метод простых итераций:")
    print(f"Приближенное решение: {root:.10f}")
    print(f"q = {q:.6f}")
    print(f"Количество итераций: {iterations}")
    print("Проверка:")
    print(f"f({root:.10f}) = {f(root):.10f}")
    print()
    root, iterations = newton_method(1.0, 1.5, max_iter=1000)

    print("Метод Ньютона:")
    print(f"Приближенное решение: {root:.10f}")
    print(f"Количество итераций: {iterations}")
    print("Проверка:")
    print(f"f({root:.10f}) = {f(root):.10f}")
    print()

    x0_newton = 0.4663
    root, iterations = newton_method(a, b, max_iter=1000, x0=x0_newton)

    print("Метод Ньютона с неверным начальным приближением:")
    print(f"Начальное приближение: x0 = {x0_newton:.4f}")
    print(f"Приближенное решение: {root:.10f}")
    print(f"Количество итераций: {iterations}")
    print("Проверка:")
    print(f"f({root:.10f}) = {f(root):.10f}")


if __name__ == "__main__":
    main()
