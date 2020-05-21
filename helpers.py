import sklearn

from MDSHelpers import *

from numpy.linalg import LinAlgError

import numpy as np
from scipy import linalg as LA
from numpy.random import normal
from functools import reduce

from scipy.linalg import norm, inv
from sklearn.manifold import MDS


def calc_abs(i, j):
    return np.abs(i - j)


def calc_abs_squared(i, j):
    return np.abs(i ** 2 - j ** 2)


def calc_squared(i, j):
    return (i - j) ** 2


def calc_C(N, cost):
    return np.array([[cost(i, j) for j in range(N)] for i in range(N)])


def calc_C_normalized(N, cost):  # ->
    C = np.array([[cost(i, j) for j in range(N)] for i in range(N)])
    return C / C.max()


def calc_d_hamm(v1, v2):
    d_hamm = 0
    for i in range(len(v1)):
        if v1[i] != v2[i]:
            d_hamm += 1
    return d_hamm


def calc_mp(p1, p2):  # returns mp of pair of points in m dimensions
    m = len(p1)
    mp = np.zeros(m)
    for i in range(0, m):
        mp[i] = (p1[i] + p2[i]) / 2
    return mp


def calc_pairwise_dists(pts):
    return np.array([[norm(pts[i] - pts[j]) for j in range(len(pts))] for i in range(len(pts))])


def calc_strain_norm(pts, C):
    return np.sqrt(
        reduce(lambda a, b: a + b,
               ((C[i][j] - norm(pts[i] - pts[j])) ** 2 for i in range(len(pts)) for j in range(len(pts)) if i != j))
        / sum([C[i][j] ** 2 for i in range(len(pts)) for j in range(len(pts))]))


def calc_strain(pts, C):
    return reduce(lambda a, b: a + b,
                  ((C[i][j] - norm(pts[i] - pts[j])) ** 2 for i in range(len(pts)) for j in range(len(pts)) if i != j))


# MDS encoding
def MDS_smacof(C, m):
    mds = MDS(m, dissimilarity='precomputed')
    return mds.fit_transform(C)


def MDS_smacof_sph(C, m, pen=60, max_iter=600):
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


# Hyperplane seperate

def refine_centroid_(clusters, m,
                     pts):  # takes k crowded clusters in m dimensions and returns w that bisects <= k of them, next cluster set after seperation
    centroids = []  # find MPs of all pairs of points
    for c in clusters:
        if len(centroids) < m:
            centroids.append(c.mean(axis=0))

    if len(centroids) < m:  # add m - k arb points to MPs until we get m points
        centroids.extend(pts[np.random.choice(len(pts), size=m - len(centroids), replace=False)])

    # perturb = normal(0, 0.01, (len(centroids), m))  # add small noise to each dimension of each point
    centroids = np.array(centroids)
    b = np.ones(m)
    try:
        w = np.linalg.solve(centroids, b)
    except LinAlgError as err:
        print('LinAlgError!!')
        print('mps: ', centroids.shape)
        print('b: ', b.shape)

    new_clusters = []  # split clusters
    for c in clusters:
        c_neg = np.array(list(filter(lambda pt: np.sign(w @ pt - 1) < 0, c)))
        c_pos = np.array(list(filter(lambda pt: np.sign(w @ pt - 1) > 0, c)))
        if len(c_neg) > 1:
            new_clusters.append(c_neg)
        if len(c_pos) > 1:
            new_clusters.append(c_pos)
    return w, new_clusters


def refine_centroid(clusters, m,
                    pts):  # takes k crowded clusters in m dimensions and returns w that bisects <= k of them, next cluster set after seperation
    centroids = []  # find MPs of all pairs of points
    for c in clusters:
        if len(centroids) < m:
            centroids.append(c.mean(axis=0))

    if len(centroids) < m:  # add m - k arb points to MPs until we get m points
        centroids.extend(pts[np.random.choice(len(pts), size=m - len(centroids), replace=False)])

    # perturb = normal(0, 0.01, (len(centroids), m))  # add small noise to each dimension of each point
    centroids = np.array(centroids)
    b = np.ones(m)
    try:
        w = np.linalg.solve(centroids, b)
    except LinAlgError as err:
        print('LinAlgError!!')
        print('mps: ', centroids.shape)
        print('b: ', b.shape)

    new_clusters = []  # split clusters
    for c in clusters:
        c_neg = np.array(list(filter(lambda pt: np.sign(w @ pt - 1) < 0, c)))
        c_pos = np.array(list(filter(lambda pt: np.sign(w @ pt - 1) > 0, c)))
        if len(c_neg) > 1:
            new_clusters.append(c_neg)
        if len(c_pos) > 1:
            new_clusters.append(c_pos)
    return w, new_clusters


# Hyperplane seperate
def refine(clusters, m,
           pts):  # takes k crowded clusters in m dimensions and returns w that bisects <= k of them, next cluster set after seperation
    mps = []  # find MPs of all pairs of points
    for c in clusters:
        for i in range(0, len(c) - 1, 2):
            if len(mps) < m:
                mps.append((c[i] + c[i + 1]) / 2)

    if len(mps) < m:  # add m - k arb points to MPs until we get m points
        mps.extend(pts[np.random.choice(len(pts), size=m - len(mps), replace=False)])

    perturb = normal(0, 0.01, (len(mps), m))  # add small noise to each dimension of each point
    mps = np.array(mps) + perturb  # connect d midpoints w a hyperplane
    b = np.ones(m)
    try:
        w = np.linalg.solve(mps, b)
    except LinAlgError as err:
        print('LinAlgError!!')
        print('mps: ', mps.shape)
        print('b: ', b.shape)

    new_clusters = []  # split clusters
    for c in clusters:
        c_neg = list(filter(lambda pt: np.sign(w @ pt - 1) < 0, c))
        c_pos = list(filter(lambda pt: np.sign(w @ pt - 1) > 0, c))
        if len(c_neg) > 1:
            new_clusters.append(c_neg)
        if len(c_pos) > 1:
            new_clusters.append(c_pos)
    return w, new_clusters


def split(clusters, w, b):
    split_clusters = []
    for c in clusters:
        c_pos = list(filter(lambda pt: np.sign(w @ pt - b) > 0, c))
        c_neg = list(filter(lambda pt: np.sign(w @ pt - b) < 0, c))
        if len(c_neg) > 1:
            split_clusters.append(c_neg)
        if len(c_pos) > 1:
            split_clusters.append(c_pos)
    return split_clusters


def variance(cluster):
    return sum([norm(pt - cluster.mean(axis=0)) for pt in cluster])


def variance_after_split(clusters, w, b):
    split_clusters = split(clusters, w, b)
    return sum([variance(cluster) for cluster in split_clusters])


def centroids(clusters):
    return np.array([cluster.mean(axis=0) for cluster in clusters])


def opt_h_normalform(clusters, vs, ps):
    min_var = np.infty
    w_opt, b_opt = None, None
    for v in vs:
        for p in ps:
            var = variance_after_split(clusters, v, np.dot(v, p))
            if var < min_var:
                min_var = var
                w_opt = v
                b_opt = np.dot(v, p)
    return w_opt, b_opt


def opt_h_cartesianform(clusters, ps, m):
    min_var = np.infty
    w_opt, b_opt = None, None
    for chosen_ps in combinations(ps, m):
        b = np.zeros(m)
        w = np.linalg.solve(np.array(chosen_ps).reshape((m, m)), b)
        var = variance_after_split(clusters, w, b)
        if var < min_var:
            min_var = var
            w_opt = w
            b_opt = b
    return w_opt, b_opt


def seperate_PCA_variance(pts):
    pts -= pts.mean(axis=0)  # Center all points
    pca = sklearn.decomposition.PCA()
    pca.fit(pts)
    clusters = [pts]
    hs = []
    while len(clusters) < m:
        w_opt, b_opt = opt_h_normalform(clusters, pca.components_, centroids(clusters))
        hs.append((w_opt, b_opt))
        clusters = split(clusters, w_opt, b_opt)
    while len(clusters) > 0:
        w_opt, b_opt = opt_h_cartesianform(clusters, centroids(clusters))
        hs.append((w_opt, b_opt))
        clusters = split(clusters, w_opt, b_opt)
    return np.array(hs)


def seperate_hyperplane(pts, refine):
    m = pts.shape[1]
    clusters = [pts]
    ws = []
    while len(clusters) > 0:
        w, clusters = refine(clusters, m, pts)  # w: found h that splits d clusters. labels: label for each class
        ws.append(w)
    return np.array(ws)


def seperate_spectral(X, C):
    N = len(X)
    # W = X @ X.T
    W = np.full((N, N), N) - C

    D = np.diag(W.sum(axis=1))
    D_neg_sqrt = np.diag(1 / np.sqrt(D.diagonal()))
    eig_vals, vs = LA.eig(D_neg_sqrt @ X @ X.T @ D_neg_sqrt)
    # eig_vals, eig_vecs = LA.eig(X @ X.T @ LA.inv(D))
    ws = [X.T @ D_neg_sqrt @ v for v in vs[1:]]
    return ws

if __name__ == "__main__":
    C = calc_C(10, calc_abs)
    pen = 100
    pts = MDS_smacof(C, 2)
    plt.title('N = 10, m = 2')
    plt.scatter(pts[:,0], pts[:,1])
    plt.show()



# plot()
