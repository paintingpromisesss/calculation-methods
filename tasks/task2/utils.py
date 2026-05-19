from math import tan, cos
EPS = 1e-12

def get_B(x1_0, x2_0):
    a11 = df1_dx1(x1_0, x2_0)
    a12 = df1_dx2(x1_0, x2_0)
    a21 = df2_dx1(x1_0, x2_0)
    a22 = df2_dx2(x1_0, x2_0)

    det = a11 * a22 - a12 * a21

    if abs(det) < 1e-14:
        raise ValueError("Якобиан слишком мал, невозможно построить матрицу B")

    b11 = a22 / det
    b12 = -a12 / det
    b21 = -a21 / det
    b22 = a11 / det

    return b11, b12, b21, b22


def phi1(x1, x2, B):
    b11, b12, b21, b22 = B
    return x1 - (b11 * f1(x1, x2) + b12 * f2(x1, x2))


def phi2(x1, x2, B):
    b11, b12, b21, b22 = B
    return x2 - (b21 * f1(x1, x2) + b22 * f2(x1, x2))


def dphi1_dx1(x1, x2, B):
    b11, b12, b21, b22 = B
    return 1 - (b11 * df1_dx1(x1, x2) + b12 * df2_dx1(x1, x2))


def dphi1_dx2(x1, x2, B):
    b11, b12, b21, b22 = B
    return -(b11 * df1_dx2(x1, x2) + b12 * df2_dx2(x1, x2))


def dphi2_dx1(x1, x2, B):
    b11, b12, b21, b22 = B
    return -(b21 * df1_dx1(x1, x2) + b22 * df2_dx1(x1, x2))


def dphi2_dx2(x1, x2, B):
    b11, b12, b21, b22 = B
    return 1 - (b21 * df1_dx2(x1, x2) + b22 * df2_dx2(x1, x2))


def get_q(x1_min, x1_max, x2_min, x2_max, B, n=1000):
    q = 0

    for i in range(n+1):
        x1 = x1_min + (x1_max - x1_min) * i / n
        
        for j in range(n + 1):
            x2 = x2_min + (x2_max - x2_min) * j / n

            row1 = abs(dphi1_dx1(x1, x2, B)) + abs(dphi1_dx2(x1, x2, B))
            row2 = abs(dphi2_dx1(x1, x2, B)) + abs(dphi2_dx2(x1, x2, B))

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

def get_x0(x1_min, x1_max, x2_min, x2_max):
    return (x1_min + x1_max) / 2, (x2_min + x2_max) / 2
