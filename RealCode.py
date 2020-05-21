from MDSHelpers import *
from helpers import *
import numpy as np
import matplotlib.pyplot as plt

class RealCode:

    def __init__(self, N, m, normalize_C=False, cost=None, MDS=None, seperate=None):
        self.N = N
        self.m = m
        self.cost = cost if cost is not None else calc_abs
        self.MDS = MDS if MDS is not None else MDS_smacof
        self.seperate = seperate if seperate is not None else seperate_hyperplane
        # self.refine = refine if refine is not None else refine_centroid
        self.C = calc_C(N, self.cost)
        if normalize_C:
            self.C = self.C / self.C.max()
        self.pts = self.MDS(self.C, self.m)
        # plot_pts(self.pts)
        # if self.seperate is seperate_spectral:
        #     self.hs = self.seperate(self.pts, self.C)
        # else:
        #     self.hs = self.seperate(self.pts, self.refine)
        self.hs = self.seperate(self.pts)
        self.code_book = np.array([self.decode_v(pt) for pt in self.pts])
        min_d_hamm = min(
            map(lambda ij: calc_d_hamm(self.code_book[ij[0]], self.code_book[ij[1]]), combinations(range(N), 2)))
        if min_d_hamm < 1:
            raise Exception('min d_hamm of code_book is < 1!')

    def print_debug(self):
        np.set_printoptions(precision=2)
        print('C: ', self.C.shape)
        print(self.C)
        pt_dists = calc_pairwise_dists(self.pts)
        print('pairwise pt dists: ', pt_dists.shape)
        print(pt_dists)
        self.plot()
        print('code_book: ', self.code_book.shape)
        print(self.code_book)
        d_hamms = self.pairwise_d_hamms()
        print('pairwise d_hamms: ', d_hamms.shape)
        print(d_hamms)

    def encode(self, i):
        return self.pts[i]

    def decode(self, pt):
        v = self.decode_v(pt)
        return np.argmin([calc_d_hamm(code_word, v) for code_word in self.code_book])

    def decode_v(self, pt):  # -> bin vec rep of pt
        return [int(np.sign(w @ pt - b)) for w, b in self.hs]

    def plot(self):
        d = 0.1
        dx = max(np.abs(self.pts[:, 0].min()), np.abs(self.pts[:, 0].max())) + d
        dy = max(np.abs(self.pts[:, 1].min()), np.abs(self.pts[:, 1].max())) + d
        plt.axis((-dx, dx, -dy, dy))
        #         plt.axis((self.pts[:, 0].min() - d,self.pts[:, 0].max() + d,self.pts[:, 1].min() - d,self.pts[:, 1].max() + d))
        plt.scatter(self.pts[:, 0], self.pts[:, 1])  # plot pts
        x0s = np.linspace(self.pts[:, 0].min() - d, self.pts[:, 0].max() + d, 2)  # plot hyper planes
        for w, b in self.hs:
            x1s = [(-w[0] * x0 + b) / w[1] for x0 in x0s]
            plt.plot(x0s, x1s)
        plt.show()

    def strain(self):
        return calc_strain(self.pts, self.C)

    def pairwise_d_hamms(self):
        return np.array(
            [[calc_d_hamm(self.code_book[i], self.code_book[j]) for j in range(self.N)] for i in range(self.N)])


def plot_pts_hs(pts, hs):
    d = 0.1
    dx = max(np.abs(pts[:, 0].min()), np.abs(pts[:, 0].max())) + d
    dy = max(np.abs(pts[:, 1].min()), np.abs(pts[:, 1].max())) + d
    plt.axis((-dx, dx, -dy, dy))
    plt.scatter(pts[:, 0], pts[:, 1])  # plot pts
    x0s = np.linspace(pts[:, 0].min() - d, pts[:, 0].max() + d, 2)  # plot hyper planes
    for w, b in hs:
        x1s = [(-w[0] * x0 + b) / w[1] for x0 in x0s]
        plt.plot(x0s, x1s)
    plt.title('separation with {} hyperplanes'.format(len(hs)))
    plt.show()

def plot_pts(pts):
    d = 0.1
    dx = max(np.abs(pts[:, 0].min()), np.abs(pts[:, 0].max())) + d
    dy = max(np.abs(pts[:, 1].min()), np.abs(pts[:, 1].max())) + d
    plt.axis((-dx, dx, -dy, dy))
    plt.scatter(pts[:, 0], pts[:, 1])  # plot pts
    N = pts.shape[0]
    plt.title('{} classes embedded in 2 dimensions'.format(N))
    plt.show()

# plot_pen_strain()
# pts = smacof_sph(calc_C(N, calc_abs), m, pen=60)
# plot_pts(pts)
# N = 10
# m = 2
# rc = RealCode(N, m, normalize_C=True, MDS=MDS_smacof_sph, seperate=seperate_hyperplane, cost=calc_abs, refine=refine_centroid)
# rc.print_debug()
