from math import tan, cos
EPS = 1e-12

B11 = 0.21249177
B12 = -0.36123602
B21 = 0.36324400
B22 = 0.38248519


def phi1(x1, x2):
    return x1 - (B11 * f1(x1, x2) + B12 * f2(x1, x2))


def phi2(x1, x2):
    return x2 - (B21 * f1(x1, x2) + B22 * f2(x1, x2))


def dphi1_dx1(x1, x2):
    return 1 - (B11 * df1_dx1(x1, x2) + B12 * df2_dx1(x1, x2))


def dphi1_dx2(x1, x2):
    return -(B11 * df1_dx2(x1, x2) + B12 * df2_dx2(x1, x2))


def dphi2_dx1(x1, x2):
    return -(B21 * df1_dx1(x1, x2) + B22 * df2_dx1(x1, x2))


def dphi2_dx2(x1, x2):
    return 1 - (B21 * df1_dx2(x1, x2) + B22 * df2_dx2(x1, x2))


def get_q(x1_min, x1_max, x2_min, x2_max, n=1000):
    q = 0

    for i in range(n+1):
        x1 = x1_min + (x1_max - x1_min) * i / n
        
        for j in range(n + 1):
            x2 = x2_min + (x2_max - x2_min) * j / n

            row1 = abs(dphi1_dx1(x1, x2)) + abs(dphi1_dx2(x1, x2))
            row2 = abs(dphi2_dx1(x1, x2)) + abs(dphi2_dx2(x1, x2))

            q = max(q, row1, row2)
    
    return q


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