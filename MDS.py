from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import norm
from sklearn.manifold import MDS
from sklearn.metrics import euclidean_distances

"""
    A, V, B are matrices used in the spherical Guttman transform mentioned by de Leeuw.
"""


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


def MDS_smacof(C, m):
    """
    Finds point embedding of cost matrix using standard SMACOF algorithm.
    :param C: Symmetric matrix describing pairwise costs between classes.
    :param m: Number of dimensions of embedding.
    :return: Point embedding.
    """
    mds = MDS(m, dissimilarity='precomputed')
    return mds.fit_transform(C)


def MDS_smacof_sph(C, m, pen=60, max_iter=600):
    """
    Finds point embedding of cost matrix when points are compelled to lie on the unit sphere.
    :param C: Symmetric matrix describing pairwise costs between classes.
    :param m: Number of dimensions of embedding.
    :param pen: Penalty factor for spherical stress term.
    :param max_iter: Number of iterations to run SMACOF for.
    :return: Point embedding.
    """
    n = len(C)
    #     X = normal(0,1,(n, m)) # init guesses on unit sphere
    #     X /= norm(X, axis=1).reshape((n,1))
    mds = MDS(m, dissimilarity='precomputed')
    X = mds.fit_transform(C)
    A = calc_A(n)
    V = calc_V(A)
    B = calc_B(X, C, A)
    X_bar = 1.0 / n * B @ X
    X_norm = X / norm(X, axis=1).reshape((n, 1))  # all p have radius 1
    for it in range(max_iter):
        # X = inv(V + pen * np.eye(n)) @ (V @ X_bar + pen * X_norm)
        X = X_norm + 1 / pen * V @ (X_bar - X_norm)
    return X


def calc_pen_strain_norm(pts):
    """
    :return: The normalized spherical stress term.
    """
    return np.sqrt(sum([(1 - norm(pt)) ** 2 for pt in pts]) / len(pts))


def calc_strain_norm(pts, C):
    """
    :return: The normalized embedding stress term.
    """
    return np.sqrt(sum([(C[i][j] - norm(pts[i] - pts[j])) ** 2 for i in range(len(pts)) for j in range(len(pts))])
                   / sum([C[i][j] ** 2 for i in range(len(pts)) for j in range(len(pts))]))


def plot_pen_strain():
    """
    Plots penalty against embedding and spherical stress terms.
    """
    norm_strains = []
    norm_pen_strains = []
    pens = np.linspace(10, 100)
    N = 10
    import helpers
    for pen in pens:
        C = helpers.calc_C(N, helpers.calc_abs)
        X = MDS_smacof_sph(C, 2, pen)
        norm_strains.append(calc_strain_norm(X, C))
        norm_pen_strains.append(calc_pen_strain_norm(X))
    plt.title('N=%d' % N)
    plt.plot(pens, norm_strains, color='blue')
    plt.plot(pens, norm_pen_strains, color='red')
    plt.legend(['normalized strain', 'normalized pen_strain'])
    plt.xlabel('pen')
    plt.ylabel('strain')
    plt.show()


if __name__ == "__main__":
    plot_pen_strain()
