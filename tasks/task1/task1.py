from utils import EPS, phi, f, df

def simple_iteration(x0, eps=EPS, max_iter=1000):
    if eps <= 0:
        raise ValueError("Эпсилон должен быть положительным")
    
    x_prev = x0

    for k in range(1, max_iter + 1):
        x_next = phi(x_prev)
        diff = abs(x_next - x_prev)

        if diff < eps:
            return x_next, k
        
        x_prev = x_next
    raise ValueError("Метод простых итераций не сошелся за максимальное количество итераций")


def newton_method(x0, eps=EPS, max_iter=1000):
    if eps <= 0:
        raise ValueError("Эпсилон должен быть положительным")
    
    x = x0
    for k in range(1, max_iter + 1):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < eps:
            raise ValueError("Производная слишком мала, метод Ньютона может не сойтись")
        
        x_next = x - fx / dfx
        dx = abs(x_next - x)

        if dx < eps:
            return x_next, k
        
        x = x_next
    raise ValueError("Метод Ньютона не сошелся за максимальное количество итераций")

def main():
    x0 = 1.1
    root, iterations = simple_iteration(x0, max_iter=1000)

    print("Метод простых итераций:")
    print(f"Приближенное решение: {root:.10f}")
    print(f"Количество итераций: {iterations}")
    print("Проверка:")
    print(f"f({root:.10f}) = {f(root):.10f}")
    print()
    root, iterations = newton_method(x0, max_iter=1000)

    print("Метод Ньютона:")
    print(f"Приближенное решение: {root:.10f}")
    print(f"Количество итераций: {iterations}")
    print("Проверка:")
    print(f"f({root:.10f}) = {f(root):.10f}")

if __name__ == "__main__":
    main()