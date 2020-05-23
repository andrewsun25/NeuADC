import numpy as np
from scipy.linalg import norm


def calc_abs(i, j):
    """
    :return: Absolute value between two classes.
    """
    return np.abs(i - j)


def calc_abs_squared(i, j):
    return np.abs(i ** 2 - j ** 2)


def calc_squared(i, j):
    return (i - j) ** 2


def calc_C(N, cost):
    """
    :return: N x N cost matrix populated by given cost function.
    """
    return np.array([[cost(i, j) for j in range(N)] for i in range(N)])


def calc_C_normalized(N, cost):
    """
    :return: (N x N) cost matrix populated by given cost function, normalized to [0,1]
    """
    C = np.array([[cost(i, j) for j in range(N)] for i in range(N)])
    return C / C.max()


def calc_d_hamm(v1, v2):
    """
    :return: Hamming distance between vectors v1 and v2.
    """
    d_hamm = 0
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            d_hamm += 1
    return d_hamm


def calc_pairwise_dists(pts):
    """
    :return: (len(pts) x len(pts)) matrix of distances between pairs of points.
    """
    return np.array([[norm(pts[i] - pts[j]) for j in range(len(pts))] for i in range(len(pts))])
