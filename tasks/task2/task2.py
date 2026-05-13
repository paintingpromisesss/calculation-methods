from utils import get_q, phi1, phi2, f1, f2, df1_dx1, df1_dx2, df2_dx1, df2_dx2, EPS


def simple_iteration(x1_0, x2_0, eps=EPS, max_iter=1000):
    if eps <= 0:
        raise ValueError("Эпсилон должен быть положительным")

    x1, x2 = x1_0, x2_0

    q = get_q(x1 - 0.1, x1 + 0.1, x2 - 0.1, x2 + 0.1)

    bad_q = False

    if q >= 1:
        print(f"Метод простых итераций не гарантирует сходимость: q = {q:.6f} >= 1")
        bad_q = True

    for k in range(1, max_iter + 1):
        x1_next = phi1(x1, x2)
        x2_next = phi2(x1, x2)

        dx = max(abs(x1_next - x1), abs(x2_next - x2))

        if bad_q:
            if dx < eps:
                return x1_next, x2_next, q, k
        else:
            error_estimate = q / (1- q) * dx
            if error_estimate < eps:
                return x1_next, x2_next, q, k

        x1, x2 = x1_next, x2_next
    raise ValueError(
        "Метод простых итераций не сошелся за максимальное количество итераций")


def newton_method(x1_0, x2_0, eps=EPS, max_iter=1000):
    if eps <= 0:
        raise ValueError("Эпсилон должен быть положительным")

    x1, x2 = x1_0, x2_0

    for k in range(1, max_iter + 1):
        a11 = df1_dx1(x1, x2)
        a12 = df1_dx2(x1, x2)
        a21 = df2_dx1(x1, x2)
        a22 = df2_dx2(x1, x2)

        b1 = -f1(x1, x2)
        b2 = -f2(x1, x2)

        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-14:
            raise ValueError(
                "Якобиан слишком мал, метод Ньютона может не сойтись")

        dx1 = (b1 * a22 - a12 * b2) / det
        dx2 = (a11 * b2 - b1 * a21) / det

        x1_next = x1 + dx1
        x2_next = x2 + dx2

        dx = max(abs(dx1), abs(dx2))

        if dx < eps:
            return x1_next, x2_next, k

        x1, x2 = x1_next, x2_next
    raise ValueError(
        "Метод Ньютона не сошелся за максимальное количество итераций")


def main():
    x1_0, x2_0 = 0.7, 0.85

    x1, x2, q, iterations = simple_iteration(x1_0, x2_0, max_iter=1000)

    print("Метод простых итераций:")
    print(f"q = {q:.6f}")
    print(f"Приближенное решение: x1 = {x1:.10f}, x2 = {x2:.10f}")
    print(f"Количество итераций: {iterations}")
    print("Проверка:")
    print(f"f1({x1:.10f}, {x2:.10f}) = {f1(x1, x2):.10f}")
    print(f"f2({x1:.10f}, {x2:.10f}) = {f2(x1, x2):.10f}")
    print()

    x1, x2, iterations = newton_method(x1_0, x2_0, max_iter=1000)
    print("Метод Ньютона:")
    print(f"Приближенное решение: x1 = {x1:.10f}, x2 = {x2:.10f}")
    print(f"Количество итераций: {iterations}")
    print("Проверка:")
    print(f"f1({x1:.10f}, {x2:.10f}) = {f1(x1, x2):.10f}")
    print(f"f2({x1:.10f}, {x2:.10f}) = {f2(x1, x2):.10f}")
    print()


if __name__ == "__main__":
    main()
