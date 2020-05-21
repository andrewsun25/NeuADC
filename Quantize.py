from itertools import combinations

import sklearn.decomposition
from numpy.linalg import LinAlgError
import numpy as np

from scipy.linalg import norm
from scipy.linalg import null_space
import matplotlib.pyplot as plt

from helpers import calc_mp


def split(clusters, w, b):
    split_clusters = []
    for c in clusters:
        # if np.all([np.sign(np.dot(pt, w) + b) > 0 for pt in c]) or np.all([np.sign(np.dot(pt, w) + b) < 0 for pt in c]):
        #     split_clusters.append(c)
        #     continue
        c_pos = list(filter(lambda pt: np.sign(np.dot(pt, w) - b) > 0, c))
        c_neg = list(filter(lambda pt: np.sign(np.dot(pt, w) - b) < 0, c))
        # if cluster lies all on one side of h, it isn't split
        if len(c_pos) == len(c) or len(c_neg) == len(c):
            split_clusters.append(c)
        else:
            if len(c_neg) > 1:
                split_clusters.append(c_neg)
            if len(c_pos) > 1:
                split_clusters.append(c_pos)
    return split_clusters


def variance(cluster):
    return sum([norm(pt - np.mean(cluster, axis=0)) for pt in cluster])


def variance_after_split(clusters, w, b):
    split_clusters = split(clusters, w, b)
    return sum([variance(cluster) for cluster in split_clusters])


def shannon_entropy(clusters, num_total_pts):
    num_pts_in_clusters = sum(len(c) for c in clusters)
    num_seperated_pts = num_total_pts - num_pts_in_clusters
    return sum(len(c) / num_total_pts * -np.log2(len(c) / num_total_pts) for c in clusters) \
           + num_seperated_pts / num_total_pts * -np.log2(1 / num_total_pts)


def centroids(clusters):
    return np.array([np.mean(cluster, axis=0) for cluster in clusters])


def medians(clusters):
    return np.array([np.median(cluster, axis=0) for cluster in clusters])


# Finds h defined by normal vector v^*, and point p^* that minimizes induced variance on clusters
def opt_h_normalform(clusters, vs, ps):
    min_var = np.infty
    w_opt, b_opt = None, None
    for v in vs:
        for p in ps:
            var = variance_after_split(clusters, v, v @ p)
            if var < min_var:
                min_var = var
                w_opt = v
                b_opt = v @ p
    return w_opt, b_opt


def opt_h_cartesianform(clusters, ps):
    m = ps.shape[1]
    min_var = np.infty
    w_opt, b_opt = None, None
    for chosen_ps in combinations(ps, m):
        # chosen_ps = np.array(chosen_ps) + np.random.normal(0, 0.01, (m, m))
        # ns = np.linalg.solve(chosen_ps, np.ones())
        # if ns.size == 0:
        #     continue
        # w = ns.reshape(m)
        try:
            w = np.linalg.solve(np.array(chosen_ps), np.ones(m))
        except LinAlgError as err:
            print(err)
            w = np.linalg.solve(np.array(chosen_ps + np.random.normal(0, 0.01, (m, m))), np.ones(m))  # add noise
        b = 1
        var = variance_after_split(clusters, w, b)
        if var < min_var:
            min_var = var
            w_opt = w
            b_opt = b
    return w_opt, b_opt


def separate_PCA_variance(pts):
    m = pts.shape[1]
    pts -= pts.mean(axis=0)  # Center pts
    pca = sklearn.decomposition.PCA()  # Get PCs of pts
    pca.fit(pts)
    clusters = [pts]
    hs = []
    # while len(clusters) < m:  # Split based on best h = n(PC, centroid)
    #     w_opt, b_opt = opt_h_normalform(clusters, pca.components_, centroids(clusters))
    #     hs.append((w_opt, b_opt))
    #     clusters = split(clusters, w_opt, b_opt)
    while len(clusters) > 0:
        if len(clusters) < m:  # find h perpendicular to PC
            w_opt, b_opt = opt_h_normalform(clusters, pca.components_, centroids(clusters))
        else:  # find h that goes optimally goes thru all centroids of clusters
            w_opt, b_opt = opt_h_cartesianform(clusters, centroids(clusters))
        hs.append((w_opt, b_opt))
        clusters = split(clusters, w_opt, b_opt)
    return np.array(hs)


# h = {x | w dot x - b = 0}
def hyperplane_intersect(pts_tointersect):
    pts_tointersect = np.array(pts_tointersect)
    m = pts_tointersect.shape[1]
    assert (len(pts_tointersect) == m)
    try:
        w = np.linalg.solve(pts_tointersect, np.ones(m))
    except LinAlgError as err:
        print(err)
        noise = np.random.normal(0, 0.01, (m, m))
        w = np.linalg.solve(pts_tointersect + noise, np.ones(m))
    w_homogenous = np.append(w, -1)
    return w / norm(w_homogenous), 1 / norm(w_homogenous)


def refine_medians(clusters, pts):
    pts = np.array(pts)
    m = pts.shape[1]
    # assert len(set([tuple(pt) for c in clusters for pt in c])) == len(pts), 'clusters must contain all pts!'
    pts_tointersect = medians(clusters)
    if len(pts_tointersect) > m:  # Find m points that result h which minimizes sum of variance over induced clusters
        max_ent = 0
        w_opt = None
        b_opt = None
        for chosen_pts in combinations(pts_tointersect, m):
            w, b = hyperplane_intersect(chosen_pts)
            ent = shannon_entropy(split(clusters, w, b), len(pts))
            if ent > max_ent:
                max_ent = ent
                w_opt = w
                b_opt = b
    elif len(pts_tointersect) < m:  # Add random pts until we have m pts to intersect with our hyperplane
        rnd_medians = []
        rnd_pts = pts[np.random.choice(len(pts), size=2 * (m - len(pts_tointersect)), replace=False)]
        for i in range(0, len(rnd_pts) - 1, 2):
            rnd_medians.append(np.median([rnd_pts[i], rnd_pts[i + 1]], axis=0))
        pts_tointersect = np.append(pts_tointersect, rnd_medians, axis=0)
        # rnd_pts = pts[np.random.choice(len(pts), size=m - len(pts_tointersect), replace=False)]
        # pts_tointersect = np.append(pts_tointersect, rnd_pts, axis=0)
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    else:
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    new_clusters = split(clusters, w_opt, b_opt)  #
    return w_opt, b_opt, new_clusters


def seperate_medians(pts):
    pts -= pts.mean(axis=0)  # Center pts
    clusters = [pts]
    hs = []
    while len(clusters) > 0:
        w, b, clusters = refine_medians(clusters, pts)  # w splits all c
        hs.append((w, b))
        # print('mag: ', norm(w))
        # for i, c in enumerate(clusters):
        #     print('cluster {}: {} pts'.format(i, len(c)))
        # print('entropy: ', shannon_entropy(clusters, len(pts)))
        # plot(pts, [hs], ['medians'])
    return hs


def refine_centroids(clusters, pts):
    pts = np.array(pts)
    m = pts.shape[1]
    # assert len(set([tuple(pt) for c in clusters for pt in c])) == len(pts), 'clusters must contain all pts!'
    pts_tointersect = centroids(clusters)
    if len(pts_tointersect) > m:  # Find m points that result h which minimizes sum of variance over induced clusters
        min_var = np.infty
        w_opt = None
        b_opt = None
        for chosen_pts in combinations(pts_tointersect, m):
            w, b = hyperplane_intersect(chosen_pts)
            var = variance_after_split(clusters, w, b)
            if var < min_var:
                min_var = var
                w_opt = w
                b_opt = b
    elif len(pts_tointersect) < m:  # Add random pts until we have m pts to intersect with our hyperplane
        rnd_pts = pts[np.random.choice(len(pts), size=m - len(pts_tointersect), replace=False)]
        pts_tointersect = np.append(pts_tointersect, rnd_pts, axis=0)
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    else:
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    new_clusters = split(clusters, w_opt, b_opt)  #
    return w_opt, b_opt, new_clusters


def seperate_centroids(pts):
    pts -= pts.mean(axis=0)  # Center pts
    clusters = [pts]
    pca = sklearn.decomposition.PCA()
    pca.fit(pts)
    clusters = split(clusters, pca.components_[0], 1)  # Split cluster using PC with largest explained variance
    hs = [(pca.components_[0], 1)]
    print('mag: ', norm(pca.components_[0]))
    for i, c in enumerate(clusters):
        print('cluster {}: {} pts'.format(i, len(c)))
    print('entropy: ', shannon_entropy(clusters, len(pts)))
    plot(pts, [hs])
    while len(clusters) > 0:
        w, b, clusters = refine_centroids(clusters, pts)  # w splits all c
        hs.append((w, b))
        print('mag: ', norm(w))
        for i, c in enumerate(clusters):
            print('cluster {}: {} pts'.format(i, len(c)))
        print('entropy: ', shannon_entropy(clusters, len(pts)))
        plot(pts, [hs])
    return hs


def seperate(pts, intersect_criterion, split_criterion, debug=False):
    pts -= pts.mean(axis=0)  # Center pts
    clusters = [pts]
    hs = []
    while len(clusters) > 0:
        w, b, clusters = refine(clusters, pts, intersect_criterion, split_criterion)  # w splits all c
        if debug:
            print('mag: ', norm(w))
            for i, c in enumerate(clusters):
                print('cluster {}: {} pts'.format(i, len(c)))
            print('{} val: {}'.format(split_criterion.__name__, split_criterion(clusters, pts)))
            plot(pts, [hs], ['{} {}'.format(split_criterion.__name__, intersect_criterion.__name__)])
        hs.append((w, b))
    return hs


def refine(clusters, pts, intersect_criterion, split_criterion):
    pts = np.array(pts)
    m = pts.shape[1]
    # assert len(set([tuple(pt) for c in clusters for pt in c])) == len(pts), 'clusters must contain all pts!'
    pts_tointersect = intersect_criterion(clusters)
    if len(pts_tointersect) > m:  # Find m points that result h which minimizes sum of variance over induced clusters
        max_val = 0
        w_opt = None
        b_opt = None
        for chosen_pts in combinations(pts_tointersect, m):
            w, b = hyperplane_intersect(chosen_pts)
            val = split_criterion(split(clusters, w, b), pts)
            if val > max_val:
                max_val = val
                w_opt = w
                b_opt = b
    elif len(pts_tointersect) < m:  # Add random pts until we have m pts to intersect with our hyperplane
        rnd_medians = []
        rnd_pts = pts[np.random.choice(len(pts), size=2 * (m - len(pts_tointersect)), replace=False)]
        for i in range(0, len(rnd_pts) - 1, 2):
            rnd_medians.append(np.median([rnd_pts[i], rnd_pts[i + 1]], axis=0))
        pts_tointersect = np.append(pts_tointersect, rnd_medians, axis=0)
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    else:
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    new_clusters = split(clusters, w_opt, b_opt)  #
    return w_opt, b_opt, new_clusters


def plot_on_axis(ax, pts, hs, title):
    # d = 0.1
    # dx = max(np.abs(pts[:, 0].min()), np.abs(pts[:, 0].max())) + d
    # dy = max(np.abs(pts[:, 1].min()), np.abs(pts[:, 1].max())) + d
    ax.set_title(title)
    d = max(np.abs(pts[:, 1].min()), np.abs(pts[:, 1].max()), np.abs(pts[:, 0].min()), np.abs(pts[:, 0].max())) + 0.1
    ax.axis((-d, d, -d, d))
    # ax.axis((-dx, dx, -dy, dy))
    ax.scatter(pts[:, 0], pts[:, 1])  # plot pts
    x0s = np.linspace(-d, d, 2)  # plot hyper planes
    for w, b in hs:  # plot hyperplanes
        x1s = [(-w[0] * x0 + b) / w[1] for x0 in x0s]
        ax.plot(x0s, x1s)


def plot(pts, hs_toplot, titles):
    if len(hs_toplot) == 1:
        plot_on_axis(plt, pts, hs_toplot[0], titles[0])
    else:
        fig, axs = plt.subplots(len(hs_toplot))
        for i, hs in enumerate(hs_toplot):
            plot_on_axis(axs[i], pts, hs, titles[i])
    plt.show()


def sum_min_margin(pts, hs):
    return sum(min(np.abs((np.dot(pt, w) - b) / norm(w)) for w, b in hs) for pt in pts)


N = 10
m = 2
pts = np.random.uniform(-2, 2, (N, m))
hs_0 = seperate(pts, refine_centroids)
hs_1 = seperate(pts, refine_medians)
print('margin centroids: ', sum_min_margin(pts, hs_0))
print('num hs: ', len(hs_0))
print('margin medians: ', sum_min_margin(pts, hs_1))
print('num hs: ', len(hs_1))
plot(pts, [hs_0, hs_1], ['centroids finished', 'medians finished'])
