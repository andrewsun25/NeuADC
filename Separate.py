import math
import random
import timeit
from collections import deque
from datetime import datetime
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize
from scipy.linalg import norm, LinAlgError

"""
    intersect_criterion are used to decide how to choose points to intersect when performing hyperplane separation.
    Includes centroids, medians, and midpoints
"""


def centroids(clusters):
    """
    :param clusters: List of crowded clusters (clusters with 2 or more points in them).
    :return: The mean point (centroid) of each cluster.
    """
    if enable_asserts:
        assert len(clusters) > 0
        for c in clusters:
            assert len(c) > 1
    return np.array([np.mean(cluster, axis=0) for cluster in clusters])


def medians(clusters):
    """
    :param clusters: List of crowded clusters (clusters with 2 or more points in them).
    :return: The median point of each cluster.
    """
    if enable_asserts:
        assert len(clusters) > 0
        for c in clusters:
            assert len(c) > 1
    return np.array([np.median(cluster, axis=0) for cluster in clusters])


def midpoints(clusters):
    """
    :param clusters: List of crowded clusters (clusters with 2 or more points in them).
    :return: The midpoint of 1 random pair of points from each cluster.
    """
    if enable_asserts:
        assert len(clusters) > 0
        for c in clusters:
            assert len(c) > 1
    mps = []
    for c in clusters:
        if enable_asserts: assert len(c) > 1
        rnd_inds = np.random.choice(len(c), size=2, replace=False)
        i, j = rnd_inds[0], rnd_inds[1]
        mps.append(np.mean([c[i], c[j]], axis=0))
    if enable_asserts: assert len(mps) == len(clusters)
    return np.array(mps)


## Helpers ##


def split(clusters, w, b):
    """
    Splits all clusters into positive/negative clusters depending on if they lie on the positive/negative side of the
    hyperplane defined by {x | w @ x - b = 0}. Clusters that lie entirely on one side of the hyperplane will not be modified.
    :param clusters: List of crowded clusters (clusters with 2 or more points in them).
    :param w: Weight vector of hyperplane.
    :param b: Bias of hyperplane.
    :return: The newly split clusters.
    """
    if enable_asserts:
        assert len(clusters) > 0 and w is not None and b is not None
        for c in clusters:
            assert len(c) > 1
            for pt in c:
                assert len(pt) > 0
    new_clusters = []
    for c in clusters:
        c_pos = list(filter(lambda pt: np.sign(np.dot(pt, w) - b) >= 0, c))
        c_neg = list(filter(lambda pt: np.sign(np.dot(pt, w) - b) < 0, c))
        # if cluster lies all on one side of h, it isn't split. So just readd it.
        if len(c_pos) == len(c) or len(c_neg) == len(c):
            new_clusters.append(c)
        else:
            if len(c_neg) > 1:
                new_clusters.append(c_neg)
            if len(c_pos) > 1:
                new_clusters.append(c_pos)
    for c in new_clusters:  # all clusters either all lie on the + side of h, or all on the - side
        if enable_asserts: assert np.all([np.sign(np.dot(pt, w) - b) >= 0 for pt in c]) or np.all(
            [np.sign(np.dot(pt, w) - b) < 0 for pt in c])
    if enable_asserts: assert sum(len(c) for c in new_clusters) <= sum(len(c) for c in clusters)
    return new_clusters


def hyperplane_intersect(pts_tointersect):
    """
    Finds the hyperplane which intersects all m points in pts_tointersect.
    The hyperplane is defined as {x | w dot x - b = 0}.
    :param pts_tointersect: Points for hyperplane to intersect.
    :return: The tuple (w,b) which defines the weight and bias of the hyperplane.
    """
    pts_tointersect = np.array(pts_tointersect)
    if enable_asserts: assert 0 < pts_tointersect.shape[0] == pts_tointersect.shape[1]
    m = pts_tointersect.shape[1]
    try:
        w = np.linalg.solve(pts_tointersect, np.ones(m))
    except LinAlgError as err:
        print('Perturbing pts because of {}'.format(err))
        noise = np.random.normal(0, 0.01, (m, m))
        w = np.linalg.solve(pts_tointersect + noise, np.ones(m))
    if enable_asserts: assert w is not None
    w_homogenous = np.append(w, -1)
    return w / norm(w_homogenous), 1 / norm(w_homogenous)


def pt_plane_distance(pt, w, b):
    """
    :return: The distance from the given pt to the given hyperplane = (w,b).
    """
    if enable_asserts: assert len(pt) > 0 and w is not None and b is not None
    return np.abs((np.dot(pt, w) - b)) / norm(w)


"""
    We want to pick the hyperplane that maximizes the value of the split_criterion.
    split_criterions include neg_variance, margin_sum, arbitrary.
"""


def shannon_entropy(clusters, w, b, pts):
    """
    :return: The shannon entropy of the new cluster set after the hyperplane (w, b) is added by treating
    {|c| / N | c in clusters} as a distribution.
    """
    if enable_asserts:
        assert len(clusters) > 0 and w is not None and b is not None
        for c in clusters:
            assert len(c) > 1
            for pt in c:
                assert len(pt) > 0
    split_clusters = split(clusters, w, b)
    num_pts_in_clusters = sum(len(c) for c in split_clusters)
    num_seperated_pts = len(pts) - num_pts_in_clusters
    return sum(len(c) / len(pts) * -np.log2(len(c) / len(pts)) for c in split_clusters) \
           + num_seperated_pts / len(pts) * -np.log2(1 / len(pts))


def neg_variance(clusters, w, b, *unused):
    """
    :return: The variance of the new cluster set as the sum of the variances of each cluster from its centroid.
    """
    if enable_asserts:
        assert len(clusters) > 0 and w is not None and b is not None
        for c in clusters:
            assert len(c) > 1
            for pt in c:
                assert len(pt) > 0
    split_clusters = split(clusters, w, b)
    return -sum(sum(norm(pt - np.mean(c, axis=0)) for pt in c) for c in split_clusters)


def margin_sum(clusters, w, b, *unused):
    """
    :return: The sum of minimum margins of each cluster split by the hyperplane.
    """
    if enable_asserts:
        assert len(clusters) > 0 and w is not None and b is not None
        for c in clusters:
            assert len(c) > 1
            for pt in c:
                assert len(pt) > 0
    sum = 0
    for c in clusters:
        # if c lies all on one side of h, it is not split by h
        if np.all([np.sign(np.dot(pt, w) - b) >= 0 for pt in c]) \
                or np.all([np.sign(np.dot(pt, w) - b) < 0 for pt in c]):
            continue
        sum += min(pt_plane_distance(pt, w, b) for pt in c)  # dist from h to closest pt in c that h is splitting
    return sum


def arbitrary(*unused):
    """
    Used for the original midpoint cluster algorithm that doesn't use any heuristics to choose hyperplanes.
    """
    return 1


## Hyperplane Separation ##


def refine(clusters, pts, intersect_criterion, split_criterion):
    """
    Finds the hyperplane that intersects up to m clusters in the cluster set using the points calculated from intersect_criterion.
    If there are > m clusters, pick the hyperplane which maximizes the split_criterion.
    :param clusters: The cluster set to split.
    :param pts: All points.
    :param intersect_criterion: The method to find which points to intersect from a cluster set.
    :param split_criterion: The method for calculating the value of a hyperplane.
    :return: The weight and bias of the optimal hyperplane and the newly split cluster set.
    """
    if enable_asserts:
        assert len(clusters) > 0 and pts.shape[0] > 0 and pts.shape[1] > 0 \
               and intersect_criterion is not None and split_criterion is not None
        for c in clusters:
            assert len(c) > 1
            for pt in c:
                assert len(pt) == pts.shape[1]
    m = pts.shape[1]
    pts_tointersect = intersect_criterion(clusters)
    if len(pts_tointersect) > m:  # Find m points that result h which minimizes sum of variance over induced clusters
        max_val = -np.infty
        w_opt = None
        b_opt = None
        for chosen_pts in combinations(pts_tointersect, m):
            chosen_pts = np.array(chosen_pts)
            w, b = hyperplane_intersect(chosen_pts)
            val = split_criterion(clusters, w, b, pts)
            if val > max_val:
                max_val = val
                w_opt = w
                b_opt = b
            if split_criterion is arbitrary:
                break
    elif len(pts_tointersect) < m:
        # Add random midpoints until we have m pts to intersect with our hyperplane.
        rnd_medians = []
        rnd_pts = pts[np.random.choice(len(pts), size=2 * (m - len(pts_tointersect)), replace=False)]
        for i in range(0, len(rnd_pts) - 1, 2):
            rnd_medians.append(np.mean([rnd_pts[i], rnd_pts[i + 1]], axis=0))
        pts_tointersect = np.append(pts_tointersect, rnd_medians, axis=0)
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    else:
        w_opt, b_opt = hyperplane_intersect(pts_tointersect)
    new_clusters = split(clusters, w_opt, b_opt)  #
    assert w_opt is not None and b_opt is not None
    return w_opt, b_opt, new_clusters


def refine_max_separating_margin(clusters, pts, *unused):
    """
    Finds the hyperplane that minimizes f.
    :param clusters: The cluster set to split.
    :param pts: All points.
    :return: The weight and bias of the optimal hyperplane and the newly split cluster set.
    """
    if enable_asserts:
        assert len(clusters) > 0 and pts.shape[0] > 0 and pts.shape[1] > 0
        for c in clusters:
            assert len(c) > 1
            for pt in c:
                assert len(pt) == pts.shape[1]

    def f(w_b, chosen_clusters):
        w = w_b[:-1]
        b = w_b[-1]
        return sum(sum((w @ pt2 - b) * (w @ pt1 - b) for pt1, pt2 in combinations(c, 2)) for c in chosen_clusters)

    def lagrangian_f(w_b_L, chosen_clusters):  # len = m + 2. Last element = lambda
        w = w_b_L[:-2]
        b = w_b_L[-2]
        L = w_b_L[-1]
        grad_f_vec = np.zeros(len(w) + 1)
        for i in range(len(w)):
            grad_f_vec[i] = sum(sum((w @ pt2 - b) * pt1[i] + (w @ pt1 - b) * pt2[i] for pt1, pt2 in combinations(c, 2))
                                for c in chosen_clusters)
        grad_f_vec[-1] = sum(sum(2 * b - w @ pt1 - w @ pt2 for pt1, pt2 in combinations(c, 2)) for c in chosen_clusters)
        L_times_grad_g_vec = L * np.append(w, 0) / norm(np.append(w, 0))  # g(w, b) = ||w|| - 1 = 0
        grad = np.zeros(w_b_L.shape)
        grad[:-1] = grad_f_vec - L_times_grad_g_vec  #
        grad[-1] = norm(w) - 1  # constraint: ||w|| - 1 = 0
        return grad

    def opt_w_b(chosen_clusters):
        min_f = np.inf  # Try random guesses for w and b and take guess that minimizes f
        w_opt = b_opt = None
        for i in range(20):
            w_init = np.random.uniform(-1, 1, m)
            w_init /= norm(w_init)
            # b_init = 0
            b_init = np.random.uniform(-1, 1)
            L_init = 0
            res = optimize.root(lambda w_b_L: lagrangian_f(w_b_L, chosen_clusters), np.hstack((w_init, b_init, L_init)))
            w_b_L = res.x
            # w_b_L = optimize.fsolve(lambda w_b_L: grad_f(w_b_L, chosen_clusters), np.hstack((w_init, b_init, L_init)))
            val = f(w_b_L[:-1], chosen_clusters)
            if val < min_f:
                min_f = val
                w_opt = w_b_L[:-2]
                b_opt = w_b_L[-2]
        return w_opt, b_opt, min_f

    m = pts.shape[1]
    min_f_all = np.inf
    w_opt = b_opt = None
    if len(clusters) > m:
        for chosen_clusters in combinations(clusters, m):
            w, b, min_f = opt_w_b(chosen_clusters)
            if min_f < min_f_all:
                min_f_all = min_f
                w_opt, b_opt = w, b
    else:
        w_opt, b_opt, min_f_all = opt_w_b(clusters)
    assert w_opt is not None and b_opt is not None
    new_clusters = split(clusters, w_opt, b_opt)
    return w_opt, b_opt, new_clusters


def hyperplane_separate(pts, intersect_criterion, split_criterion, refine=refine, debug=False, save_pts=False):
    """
    Continually splits (AKA refines) the point set into smaller clusters.
    :param pts: All points to split.
    :param intersect_criterion: Method for calculating which points to intersect.
    :param split_criterion: Method for choosing which hyperplane to select
    :param refine: Which refine function to use.
    :return: List of hyperplanes that separate all pairs of pts.
    """
    if enable_asserts: assert pts.shape[0] > 0 and pts.shape[1] > 0
    if save_pts:
        time = datetime.now().strftime("%H:%M:%S")
        np.save('tmp/pts_{}'.format(time), pts)
    clusters = [pts]
    hs = []
    while len(clusters) > 0:
        old_clusters = clusters
        w, b, clusters = refine(clusters, pts, intersect_criterion, split_criterion)  # w splits all c
        hs.append((w, b))
        if debug:
            ic_name = intersect_criterion.__name__ if intersect_criterion is not None else ''
            sc_name = split_criterion.__name__ if split_criterion is not None else ''
            plot(pts, [hs], ['{} {}'.format(ic_name, sc_name)])
            print('HYPERPLANE #: {}'.format(len(hs)))
            print('mag: ', norm(w))
            print('w: ', w)
            print('b: ', b)
            for i, c in enumerate(clusters):
                print('cluster {}: {} pts'.format(i, len(c)))
            if split_criterion is not None:
                print('{} val: {}'.format(sc_name, split_criterion(old_clusters, w, b, pts)))
    if enable_asserts:
        for pt_i, pt_j in combinations(pts, 2):  # all pairs need to be separated by at least 1 hyperplane
            assert np.any([(np.dot(w, pt_i) - b) * (np.dot(w, pt_j) - b) < 0 for w, b in hs])
    return hs


def d_hamm(pt_1, pt_2, hs):
    """
    :return: The number of hyperplanes in hs that split pt_1 and pt_2.
    """
    return int(np.sum([1 for w, b in hs if (w @ pt_1 - b) * (w @ pt_2 - b) < 0]))


def hyperplane_separate_D_queue(pts, D, debug=False):
    """
    Separates all pts until all pairs are at least D apart. All pairs of points are initially put into a queue and
    visited one at a time until all pairs are D apart.
    :return: List of separating hyperplanes.
    """
    if enable_asserts: assert pts.shape[0] > 0 and pts.shape[1] > 0 and D > 0
    m = pts.shape[1]
    pt_pairs = list(combinations(pts, 2))
    random.shuffle(pt_pairs)
    pair_queue = deque(pt_pairs)
    hs = []
    while len(pair_queue) > 0:
        pts_tointersect = []
        for j in range(m):
            top_pair = pair_queue.popleft()
            pt_1, pt_2 = top_pair[0], top_pair[1]
            pts_tointersect.append((pt_1 + pt_2) / 2)
            if debug: print('adding pt {} to intersect'.format((pt_1 + pt_2) / 2))
            if d_hamm(pt_1, pt_2, hs) < D - 1:
                pair_queue.append(top_pair)
            if len(pair_queue) == 0:
                break
        if len(pts_tointersect) < m:
            rnd_pairs = [pt_pairs[i] for i in
                         np.random.choice(len(pt_pairs), size=m - len(pts_tointersect), replace=False)]
            for pt_1, pt_2 in rnd_pairs:
                pts_tointersect.append((pt_1 + pt_2) / 2)
        w, b = hyperplane_intersect(pts_tointersect)
        hs.append((w, b))
        if debug:
            print('HYPERPLANE #: {}'.format(len(hs)))
            print('mag: ', norm(w))
            print('w: ', w)
            print('b: ', b)
            print('queue length: ', len(pair_queue))
            plot(pts, [hs], [''])
    # Removes hyperplane if doing so won't cause any pair's d_hamm to dip below D
    i = 0
    while i < len(hs):
        hs_except_i = hs[:i] + hs[i + 1:]
        if np.all([d_hamm(pt_1, pt_2, hs_except_i) >= D for pt_1, pt_2 in pt_pairs]):
            hs.pop(i)
        i += 1
    print('num hyperplanes: {}'.format(len(hs)))
    if enable_asserts:
        for pt_1, pt_2 in combinations(pts, 2):  # all pairs need to be separated by at least D hyperplanes
            assert d_hamm(pt_1, pt_2, hs) >= D
    return hs


def hyperplane_separate_D_buckets(pts, D, debug=False):
    """
    Separates all pts until all pairs are at least D apart. All pairs of points that are i separated are put into bucket
    i. Lowest buckets are visited first at each iteration until all pairs are D separated.
    :return: List of separating hyperplanes.
    """
    if enable_asserts: assert pts.shape[0] > 0 and pts.shape[1] > 0 and D > 0
    m = pts.shape[1]
    pt_pairs = list(combinations(pts, 2))
    random.shuffle(pt_pairs)
    clusters = [[] for _ in range(D + 1)]
    for pair in pt_pairs:
        clusters[0].append(pair)
    hs = []
    while len(clusters[D]) < len(pt_pairs):  # while last cluster (with d_hamm D) doesn't have all pairs
        pts_tointersect = []
        i = 0
        while len(pts_tointersect) < m and i < len(clusters):
            curr_cluster = clusters[i]
            while len(curr_cluster) > 0 and len(
                    pts_tointersect) < m:  # choose random points from curr_cluster and remove them
                num_chosen = min(len(curr_cluster), m - len(pts_tointersect))
                chosen_indices = np.random.choice(len(curr_cluster), size=num_chosen, replace=False)
                for i in chosen_indices:
                    pt_1, pt_2 = curr_cluster[i]
                    if debug: print('adding pt {} to intersect'.format((pt_1 + pt_2) / 2))
                    pts_tointersect.append((pt_1 + pt_2) / 2)
                curr_cluster = [curr_cluster[i] for i in range(len(curr_cluster)) if
                                i not in chosen_indices]  # remove chosen points from curr_cluster
            i += 1
        if len(pts_tointersect) < m:
            rnd_pairs = [pt_pairs[i] for i in
                         np.random.choice(len(pt_pairs), size=m - len(pts_tointersect), replace=False)]
            for pt_1, pt_2 in rnd_pairs:
                pts_tointersect.append((pt_1 + pt_2) / 2)
        w, b = hyperplane_intersect(pts_tointersect)
        hs.append((w, b))
        new_clusters = [[] for _ in range(D + 1)]
        for pt_1, pt_2 in pt_pairs:
            d = min(d_hamm(pt_1, pt_2, hs), D)
            new_clusters[d].append((pt_1, pt_2))
        if enable_asserts: assert sum(len(c) for c in new_clusters) == len(pt_pairs)
        clusters = new_clusters
        if debug:
            print('HYPERPLANE #: {}'.format(len(hs)))
            print('mag: ', norm(w))
            print('w: ', w)
            print('b: ', b)
            plot(pts, [hs], [''])
    # Removes hyperplane if doing so won't cause any pair's d_hamm to dip below D
    i = 0
    while i < len(hs):
        hs_except_i = hs[:i] + hs[i + 1:]
        if np.all([d_hamm(pt_1, pt_2, hs_except_i) >= D for pt_1, pt_2 in pt_pairs]):
            hs.pop(i)
        i += 1
    print('num hyperplanes: {}'.format(len(hs)))
    if enable_asserts:
        for pt_1, pt_2 in combinations(pts, 2):  # all pairs need to be separated by at least D hyperplanes
            assert d_hamm(pt_1, pt_2, hs) >= D
    return hs


## Misc ##

def plot_on_axis(ax, pts, hs, title):
    """
    Plots points and its corresponding hyperplane separation in 2D.
    :param ax: Axis to plot on.
    :param pts: Points to plot.
    :param hs: Hyperplane separation of pts to plot.
    :param title: Title of graph.
    """
    try:
        ax.set_title(title)
    except AttributeError:
        plt.title(title)
    d = max(np.abs(pts[:, 1].min()), np.abs(pts[:, 1].max()), np.abs(pts[:, 0].min()),
            np.abs(pts[:, 0].max())) + 0.1
    ax.axis((-d, d, -d, d))
    ax.scatter(pts[:, 0], pts[:, 1])  # plot pts
    x0s = np.linspace(-d, d, 2)
    for w, b in hs:  # plot hyperplanes
        x1s = [(-w[0] * x0 + b) / w[1] for x0 in x0s]
        ax.plot(x0s, x1s)


def plot(pts, hs_toplot, titles):
    """
    Plots a list of hyperplane separations for the same point set.
    """
    if enable_asserts:
        assert len(hs_toplot) > 0
        assert len(hs_toplot) == len(titles)
    if len(hs_toplot) == 1:
        plot_on_axis(plt, pts, hs_toplot[0], titles[0])
    else:
        if len(hs_toplot) % 2 == 0:
            nrows = int(len(hs_toplot) / 2)
            fig, axs = plt.subplots(nrows=nrows, ncols=int(len(hs_toplot) / nrows))
            axs = axs.ravel()
        else:
            fig, axs = plt.subplots(ncols=len(hs_toplot))
        for i, hs in enumerate(hs_toplot):
            plot_on_axis(axs[i], pts, hs, titles[i])
    plt.show()


def margin(hs, pts):
    """
    :return: The margin of the hyperplane separation on point set.
    """
    margin_sum = 0
    for pt_i, pt_j in combinations(pts, 2):
        max_margin = 0
        for w, b in hs:
            if (np.dot(pt_i, w) - b) * (np.dot(pt_j, w) - b) < 0:
                max_margin = max(max_margin, min(pt_plane_distance(pt_i, w, b), pt_plane_distance(pt_j, w, b)))
        margin_sum += max_margin
    return margin_sum


def plot_m_vs_num_hyperplanes(N, ms, num_iters_per_m, algos_toplot):
    """
    Plots m against the size of the hyperplane separations found using specified algos.
    :param N: Number of points.
    :param ms: List of m values to plot.
    :param num_iters_per_m: Number of trials ot average for each value of m.
    :param algos_toplot: List of algos to generate hyperplane separations.
    """
    algo_plots = np.zeros((len(algos_toplot), len(ms)))
    for i, (ic, sc) in enumerate(algos_toplot):
        for j, m in enumerate(ms):
            print('m: {}. algo: {} {}'.format(m, ic.__name__, sc.__name__))
            algo_plots[i][j] = np.mean(
                [len(hyperplane_separate(np.random.uniform(-1, 1, (N, m)), ic, sc, debug=False)) for _ in
                 range(num_iters_per_m)])
    for i in range(len(algo_plots)):
        plt.plot(ms, algo_plots[i])
    plt.xlabel('m')
    plt.ylabel('num hyperplanes')
    plt.title('num hyperplanes to completely separate {} points in m dimensions'.format(N))
    plt.legend(['{} {}'.format(ic.__name__, sc.__name__) for ic, sc in algos_toplot])
    plt.show()


def plot_m_vs_margin(N, ms, num_iters_per_m, algos_toplot):
    """
    Plots m against the margin of the hyperplane separations found using specified algos.
    :param N: Number of points.
    :param ms: List of m values to plot.
    :param num_iters_per_m: Number of trials ot average for each value of m.
    :param algos_toplot: List of algos to generate hyperplane separations.
    """
    algo_plots = np.zeros((len(algos_toplot), len(ms)))
    for i, (ic, sc) in enumerate(algos_toplot):
        for j, m in enumerate(ms):
            print('m: {}. algo: {} {}'.format(m, ic.__name__, sc.__name__))
            for _ in range(num_iters_per_m):
                start_time = timeit.default_timer()
                pts = np.random.uniform(-1, 1, (N, m))
                hs = hyperplane_separate(pts, ic, sc)
                algo_plots[i][j] += margin(hs, pts)
                time_per_iter = timeit.default_timer() - start_time
                print('~{} seconds left'.format((len(algos_toplot) - i) * (len(ms) - j) * time_per_iter))
            algo_plots[i][j] /= num_iters_per_m
    for i in range(len(algo_plots)):
        plt.plot(ms, algo_plots[i])
    plt.xlabel('m')
    plt.ylabel('sum max margin')
    plt.title('sum_(p_i, p_j) max_(splitting h) min(d(p_i, h), d(p_j, h)) for {} points'.format(N))
    plt.legend(['{} {}'.format(ic.__name__, sc.__name__) for ic, sc in algos_toplot])
    plt.show()


enable_asserts = True

if __name__ == "__main__":
    # pts = np.array([[0, 1], [0.7, 0.2], [0.2, -0.7], [-0.2, -0.7], [-0.7, 0.2]], dtype=float)  # pentagon
    #
    # hs0 = hyperplane_separate(pts, midpoints, arbitrary)
    # hs1 = hyperplane_separate_D_queue(pts, 3)
    # hs2 = hyperplane_separate_D_buckets(pts, 3)
    # plot(pts, [hs0, hs1, hs2], ['midpoint cluster', 'queue D separate', 'buckets D separate'])
    N = 40
    ms = range(2, 20, 2)
    num_iters_per_m = 10
    intersect_criterions = [midpoints, centroids]
    split_criterions = [arbitrary, margin_sum, shannon_entropy]
    algos = [(ic, sc) for ic in intersect_criterions for sc in split_criterions]
    plot_m_vs_num_hyperplanes(N, ms, 2, algos)
