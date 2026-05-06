from math import log
EPS = 1e-12


def phi(x):
    if x <= -2:
        raise ValueError("x должен быть больше -2")
    value = log(x+2) + 0.5

    if value < 0:
        raise ValueError("phi(x) должен быть неотрицательным")

    return value**0.25


def dphi(x):
    if x <= -2:
        raise ValueError("x должен быть больше -2")
    value = log(x+2) + 0.5

    if value < 0:
        raise ValueError("phi(x) должен быть неотрицательным")

    return 1 / (4 * (x + 2) * value ** 0.75)


def f(x):
    if x <= -2:
        raise ValueError("x должен быть больше -2")
    return log(x+2) - x**4 + 0.5


def df(x):
    if x <= -2:
        raise ValueError("x должен быть больше -2")
    return 1/(x+2) - 4*x**3

def ddf(x):
    if x <= -2:
        raise ValueError("x должен быть больше -2")
    return -1/(x+2)**2 - 12*x**2

def get_x0_newton(a, b):
    fa = f(a)
    fb = f(b)

    ddf_a = ddf(a)
    ddf_b = ddf(b)

    if fa * ddf_a > 0:
        return a
    elif fb * ddf_b > 0:
        return b
    else:
        raise ValueError("Невозможно выбрать начальное приближение для метода Ньютона")
    
def get_x0_simple_iteration(a, b):
    return (a + b) / 2