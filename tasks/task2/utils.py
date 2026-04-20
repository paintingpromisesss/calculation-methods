from math import atan, sqrt, tan, cos
EPS = 1e-12


def phi1(x1, x2):
    return atan(x2)


def phi2(x1, x2):
    value = 1 + x1 - 2 * x1**2
    if value < 0:
        raise ValueError(
            "Значение под корнем отрицательное, метод простых итераций не применим")

    return sqrt(value)


def f1(x1, x2):
    return 2 * x1**2 - x1 + x2**2 - 1


def f2(x1, x2):
    return x2 - tan(x1)


def df1_dx1(x1, x2):
    return 4 * x1 - 1


def df1_dx2(x1, x2):
    return 2 * x2


def df2_dx1(x1, x2):
    return - (1 / cos(x1)**2)

def df2_dx2(x1, x2):
    return 1