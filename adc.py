import numpy as np

import matplotlib.pyplot as plt
from NeuADC import NeuADC, simple_model
from Quantize_new import seperate_hyperplane, plot_on_axis
from RealCode import RealCode, plot_pts, plot_pts_hs
from helpers import calc_C, calc_abs, MDS_smacof, MDS_smacof_sph


def label_linear(signals):
    sig_range = signals.max() - signals.min() + 0.001 # 1
    levels = [int(sig / sig_range * len(signals)) for sig in signals]
    return np.array(levels)

if __name__ == "__main__":
    num_pts = 10
    v_in_max = 1.0
    signals = np.linspace(0, v_in_max, num_pts)
    levels = label_linear(signals)
    N = len(levels)
    # m = int(np.sqrt(N))
    m = 2
    C = calc_C(N, calc_abs)
    encoded_pts = MDS_smacof(C, m)
    # plot_pts(encoded_pts)

    hs = seperate_hyperplane(encoded_pts)
    plot_on_axis(plt, encoded_pts, hs, 'hyperplane separation')
    plt.show()
    # rc = RealCode(N, m, cost=calc_abs, MDS=MDS_smacof, seperate=seperate_hyperplane)
    # model = NeuADC(rc, simple_model(signals.shape))

    # MDS encoding
