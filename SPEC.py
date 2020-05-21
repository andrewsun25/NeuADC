import tensorflow as tf
from numpy.random import normal
from helpers import *
from tensorflow.keras.losses import KLDivergence

def kl_divergence(S, ws):  # returns KL div between 2 probability distributions
    # X = Categorical(probs=x)
    # Y = Categorical(probs=y)
    # return kl_divergence(X, Y)
    W = calc_W_unique(pts, ws)
    return tf.tensordot(S, tf.math.log(tf.divide(S, W)), axes=[[0],[0]])
    # return tf.reduce_sum(tf.multiply(S, tf.math.log(tf.divide(S, W))))
    # return dist_i.T @ np.log(dist_i / dist_j)


def RBF_sim(pt_i, pt_j, ep):
    return np.exp(-norm(pt_i - pt_j) ** 2 / ep ** 2)

def dot(tensor_i, tensor_j):
    return tf.reduce_sum(tf.multiply(tensor_i, tensor_j))

def d_hamm(pt_i, pt_j, ws):
    d = 0
    for w in tf.convert_to_tensor(ws):
        if tf.sigmoid(dot(pt_i, w)) != tf.sigmoid(dot(pt_j, w)):
            d += 1
    return d


def calc_S(pts):
    S = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            S[i][j] = RBF_sim(pts[i], pts[j], ep)
    S /= S.sum()
    return S


def calc_S_unique(pts):
    S = []
    for i, j in combinations(range(N), 2):
        S.append(RBF_sim(pts[i], pts[j], ep))
    S = np.array(S)
    S /= S.sum()
    return tf.convert_to_tensor(S)


def calc_W_unique(pts, ws):
    W = []
    for i, j in combinations(range(N), 2):
        W.append(np.exp(-L * d_hamm(pts[i], pts[j], ws)))
    W = np.array(W)
    W /= W.sum()
    return tf.convert_to_tensor(W)


def calc_W(pts, ws):
    W = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            W[i][j] = np.exp(-L * d_hamm(pts[i], pts[j], ws))
    W /= W.sum()
    return W


N = 10
m = 2
T = 10
ep = 1
L = 1
mds = MDS(m, dissimilarity='precomputed')
C = calc_C(N, calc_abs)
pts = mds.fit_transform(C)
pts = tf.convert_to_tensor(np.hstack((np.ones((N, 1)), pts)))
ws = tf.Variable(np.hstack((np.zeros((N, 1)), normal(0, 1, (N, m)))))
S = calc_S_unique(pts)
W = calc_W_unique(pts, ws)

iter = 0
max_iters = 300
lr = tf.constant(0.01)

while iter < max_iters:
    with tf.GradientTape() as g:
        g.watch(ws)
        kl = kl_divergence(S, ws)
        tf.print(kl)
    grad = g.gradient(kl, ws)
    ws.assign_sub(grad * lr)


# def cut(S, b):
#     val = 0
#     for i, j in combinations(N, 2):
#         if b(i) != b(j):
#            val += S[i][j]
#     return val
#
# def cut2(H, b):
#     val = 0
#     for i, j in combinations(N, 2):
#         if b(i) != b(j):
#             val += np.exp(-H(i, j))
#     return val
