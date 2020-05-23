from MDS import calc_strain_norm, MDS_smacof
from Separate import *
from helpers import *


class RealCode:

    def __init__(self, N, m, normalize_C=False, cost=None, MDS=None, separate=None):
        """
        An object that encodes/decodes between classes and Euclidean space.
        :param N: Number of classes
        :param m: Number of dimensions
        :param normalize_C: Whether or not costs should be normalized to [0, 1]
        :param cost: Pairwise cost function.
        :param MDS: MDS algorithm to use.
        :param separate: Hyperplane separation algorithm to use.
        """
        self.N = N
        self.m = m
        self.cost = cost if cost is not None else calc_abs
        self.MDS = MDS if MDS is not None else MDS_smacof
        self.seperate = separate if separate is not None else lambda pts: hyperplane_separate(pts, midpoints, arbitrary)
        self.C = calc_C(N, self.cost)
        if normalize_C:
            self.C = self.C / self.C.max()
        self.pts = self.MDS(self.C, self.m)
        self.hs = self.seperate(self.pts)
        self.code_book = np.array([self.decode_v(pt) for pt in self.pts])
        min_d_hamm = min(
            map(lambda ij: calc_d_hamm(self.code_book[ij[0]], self.code_book[ij[1]]), combinations(range(N), 2)))
        if min_d_hamm < 1:
            raise Exception('min d_hamm of code_book is < 1!')

    def print_debug(self):
        """
        Prints/plots the important characteristics of the RealCode.
        """
        np.set_printoptions(precision=2)
        print('C: ', self.C.shape)
        print(self.C)
        pt_dists = calc_pairwise_dists(self.pts)
        print('pairwise pt dists: ', pt_dists.shape)
        print(pt_dists)
        print('code_book: ', self.code_book.shape)
        print(self.code_book)
        d_hamms = self.pairwise_d_hamms()
        print('pairwise d_hamms: ', d_hamms.shape)
        print(d_hamms)
        print('number of hyperplanes in separation: ', len(self.hs))
        self.plot()

    def encode(self, i):
        """
        :return: Encoding of class i as a point in Euclidean space.
        """
        return self.pts[i]

    def decode(self, pt):
        """
        :return: Decoding of an Euclidean point back into a class using hyperplane separation.
        """
        v = self.decode_v(pt)
        return np.argmin([calc_d_hamm(code_word, v) for code_word in self.code_book])

    def decode_v(self, pt):  # -> bin vec rep of pt
        """
        :return: Decoding of an Euclidean point back into a binary class vector.
        """
        return [int(np.sign(w @ pt - b)) for w, b in self.hs]

    def plot(self):
        """
        Plots class points and their hyperplane separation.
        """
        d = 0.1
        dx = max(np.abs(self.pts[:, 0].min()), np.abs(self.pts[:, 0].max())) + d
        dy = max(np.abs(self.pts[:, 1].min()), np.abs(self.pts[:, 1].max())) + d
        plt.axis((-dx, dx, -dy, dy))
        plt.scatter(self.pts[:, 0], self.pts[:, 1])  # plot pts
        x0s = np.linspace(self.pts[:, 0].min() - d, self.pts[:, 0].max() + d, 2)  # plot hyper planes
        for w, b in self.hs:
            x1s = [(-w[0] * x0 + b) / w[1] for x0 in x0s]
            plt.plot(x0s, x1s)
        plt.show()

    def strain(self):
        """
        :return: The normalized strain of points found with MDS algorithm.
        """
        return calc_strain_norm(self.pts, self.C)

    def pairwise_d_hamms(self):
        """
        :return: The pairwise hamming distances of all classes in the code_book.
        """
        return np.array(
            [[calc_d_hamm(self.code_book[i], self.code_book[j]) for j in range(self.N)] for i in range(self.N)])


if __name__ == '__main__':
    N = 10
    m = 2
    rc = RealCode(N, m, normalize_C=True, MDS=MDS_smacof)
    rc.print_debug()
