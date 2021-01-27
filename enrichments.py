from functools import partial
from scipy import stats
import numpy as np
from scipy.optimize import fsolve, minimize, minimize_scalar

# Asymptotic tests based on normal approximations
# doi:10.1002/bimj.200710403

def R_from_z(k2, n2, k1, n1, z):
    '''Analytical solution for getting R from a fixed z value. Uses sqrt method'''
    a = np.power(z, 2) / 4 - (k2 + 3/8)
    b = 2 * np.sqrt(k1 + 3/8) * np.sqrt(k2 + 3/8)
    c = np.power(z, 2) / 4 - (k1 + 3/8)
    x = (-b - np.sign(z) * np.sqrt(np.clip(np.power(b, 2) - 4*a*c, 0, np.inf))) / (2*a)
    return np.power(x, 2) * n2/n1

def R_range(k1, n1, k2, n2, zstat=2):
    return (
        R_from_z(k1, n1, k2, n2, 0),
        R_from_z(k1, n1, k2, n2, -zstat),
        R_from_z(k1, n1, k2, n2, +zstat)
    )

def R(cts, zstat=0):
    k1, n1, k2, n2 = cts
    return R_from_z(k2, n2, k1, n1, zstat)

def R_lb(cts):
    return R(cts, -2)

def R_ub(cts):
    return R(cts, +2)

def R_ranges(k1s, n1s, k2s, n2s, zstat=2, **kwargs):
    return R_range(k1s, n1s, k2s, n2s, zstat)

if __name__ == '__main__':
    c1s = np.array([10, 20, 10, 40]*10000)
    e1s = np.ones(c1s.shape) * c1s.sum()

    c2s = np.array([20, 10, 10, 40]*10000)
    e2s = np.ones(c2s.shape) * c1s.sum()

    (Rs, Rs_lb, Rs_ub) = R_ranges(c1s, e1s, c2s, e2s)
    print(Rs[-1])
    print(Rs_lb[-1])
    print(Rs_ub[-1])
