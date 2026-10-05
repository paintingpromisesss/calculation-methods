import math


EPS = 1e-12

# Границы области: x, y из [0, pi/2]
A = 0.0
B = math.pi / 2

# Точность остановки итераций
TOL = 1e-9

# Параметр верхней релаксации
OMEGA = 1.5

# Сетки для исследования сходимости (h = (pi/2)/N)
N_VALUES = (8, 16, 32)


def exact_solution(x, y):
    # Точное решение U(x, y) = exp(-x-y) cos(x) cos(y)
    return math.exp(-x - y) * math.cos(x) * math.cos(y)


def boundary_left(y):
    # Граничное условие u(0, y) = exp(-y) cos(y)
    return math.exp(-y) * math.cos(y)


def boundary_right(y):
    # Граничное условие u(pi/2, y) = 0
    return 0.0


def boundary_bottom(x):
    # Граничное условие u(x, 0) = exp(-x) cos(x)
    return math.exp(-x) * math.cos(x)


def boundary_top(x):
    # Граничное условие u(x, pi/2) = 0
    return 0.0


def get_grid(n):
    if n < 2:
        raise ValueError("Число разбиений сетки должно быть не меньше 2")

    # Равномерная сетка с шагом h по обеим переменным
    h = (B - A) / n
    xs = [A + i * h for i in range(n + 1)]
    ys = [A + j * h for j in range(n + 1)]
    return h, xs, ys


def set_boundary(u, xs, ys, n):
    # Граничные условия первого рода (Дирихле)
    for j in range(n + 1):
        u[j][0] = boundary_left(ys[j])
        u[j][n] = boundary_right(ys[j])

    for i in range(n + 1):
        u[0][i] = boundary_bottom(xs[i])
        u[n][i] = boundary_top(xs[i])


def initial_guess(u, xs, ys, n):
    # Нулевое приближение: линейная интерполяция между левой и правой
    # границами вдоль линии y = const (как в разд. 5.3.2 методички)
    width = B - A
    for j in range(1, n):
        for i in range(1, n):
            u[j][i] = boundary_left(ys[j]) * (1 - xs[i] / width)


def get_max_error(u, xs, ys, n):
    # Максимальная погрешность |U(x_i, y_j) - u_{i,j}| по всей сетке
    error = 0.0
    for j in range(n + 1):
        for i in range(n + 1):
            diff = abs(exact_solution(xs[i], ys[j]) - u[j][i])
            if diff > error:
                error = diff
    return error


def convergence_order(err_h, err_2h):
    # Порядок сходимости p = log2(err(2h)/err(h))
    if err_h < EPS or err_2h < EPS:
        raise ValueError("Погрешность слишком мала для оценки порядка")

    return math.log(err_2h / err_h) / math.log(2)


def print_solution_table(title, u, xs, ys, n):
    # Таблица решения в сравнении с точным (часть строк сетки)
    print(title)
    print()

    print(
        f"{'i':<4}"
        f"{'j':<4}"
        f"{'x':>14}"
        f"{'y':>14}"
        f"{'u числ.':>14}"
        f"{'u точн.':>14}"
        f"{'погрешность':>14}"
    )

    step = max(1, n // 4)
    for j in range(0, n + 1, step):
        for i in range(0, n + 1, step):
            u_exact = exact_solution(xs[i], ys[j])
            error = abs(u_exact - u[j][i])
            print(
                f"{i:<4}"
                f"{j:<4}"
                f"{xs[i]:>14.6f}"
                f"{ys[j]:>14.6f}"
                f"{u[j][i]:>14.6f}"
                f"{u_exact:>14.6f}"
                f"{error:>14.6e}"
            )

    print(f"Максимальная погрешность: {get_max_error(u, xs, ys, n):.6e}")
    print()


def print_method_table(title, rows):
    # Сводная таблица: сетка, итерации, достигнутая точность, погрешность
    print(title)
    print()

    print(
        f"{'N':<6}"
        f"{'h':>14}"
        f"{'итераций':>12}"
        f"{'достигнутая точность':>22}"
        f"{'погрешность':>14}"
    )

    for row in rows:
        n, h, iterations, diff, error = row
        print(
            f"{n:<6}"
            f"{h:>14.6f}"
            f"{iterations:>12}"
            f"{diff:>20.3e}"
            f"{error:>14.3e}"
        )

    print()


def print_order_table(title, rows):
    # Таблица порядка сходимости
    print(title)
    print()

    print(
        f"{'N':<6}"
        f"{'h':>14}"
        f"{'погрешность':>14}"
        f"{'порядок p':>14}"
    )

    for row in rows:
        n, h, error, order = row
        order_str = f"{order:.2f}" if order is not None else "-"
        print(
            f"{n:<6}"
            f"{h:>14.6f}"
            f"{error:>14.3e}"
            f"{order_str:>14}"
        )

    print()
