from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from sklearn.metrics import euclidean_distances

def calc_A(n):
    A = np.zeros((n, n, n, n))
    for i, j in combinations(range(n), 2):
        for r in range(n):
            for c in range(n):
                if r == c and (c == i or c == j):
                    A[i][j][r][c] = 1
                elif r == i and c == j or r == j and c == i:
                    A[i][j][r][c] = -1
                else:
                    A[i][j][r][c] = 0
    return A

def calc_V(A):
    n = len(A)
    V = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        V += A[i][j]
    return V


def calc_B(X, C, A):
    n = len(C)
    dis = euclidean_distances(X)
    B = np.zeros((n, n))
    for i, j in combinations(range(n), 2):
        B += C[i][j] / dis[i][j] * A[i][j]
    return B

def calc_pen_strain(pts):
    return np.sqrt(sum([(1 - norm(pt)) ** 2 for pt in pts]) / (2 * len(pts)))


def plot_pen_strain():
    pen_strain_norm = []
    pen_pen_strain = []
    pens = np.linspace(10, 100)
    N = 10
    for pen in pens:
        import helpers
        C = helpers.calc_C(N, helpers.calc_abs)
        X = helpers.MDS_smacof_sph(C, 2, pen)
        pen_strain_norm.append(helpers.calc_strain_norm(X, C))
        pen_pen_strain.append(calc_pen_strain(X))
    plt.title('N=%d' % N)
    plt.plot(pens, pen_strain_norm, color='blue')
    plt.plot(pens, pen_pen_strain, color='red')
    plt.legend(['normalized strain', 'normalized pen_strain'])
    plt.xlabel('pen')
    plt.ylabel('strain')
    plt.show()


def plot_m_strain():
    m_strain_norm = []
    m_pen_strain = []
    ms = range(2, 8)
    N = 10
    for m in ms:
        import helpers
        C = helpers.calc_C(N, helpers.calc_abs)
        X = helpers.MDS_smacof_sph(C, m, 27)
        m_strain_norm.append(helpers.calc_strain_norm(X, C))
        m_pen_strain.append(calc_pen_strain(X))
    print(m_strain_norm)
    plt.title('N=%d' % N)
    plt.plot(ms, m_strain_norm, color='blue')
    plt.plot(ms, m_pen_strain, color='red')
    plt.legend(['strain', 'pen_strain'])
    plt.xlabel('m')
    plt.ylabel('strain')
    plt.show()

# plot_pen_strain()
# plot_m_strain()
# np.set_printoptions(precision=2)
# C = calc_C(10, calc_abs)
# pts = smacof_sph(C, 2, 1000)
# plt.scatter(pts[:,0], pts[:,1])
# plt.show()
