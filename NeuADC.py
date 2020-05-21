import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense, Flatten
from keras.models import Sequential
from scipy.linalg import norm


def euc_dist_loss(y_true, y_pred):
    import keras.backend as K
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))

def simple_model(input_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
    ])
    return model

def regression_model(input_shape):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(1)
    ])
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['mae', 'mse'])
    return model

class NeuADC:
    def __init__(self, real_code, model, loss=euc_dist_loss):
        self.real_code = real_code
        self.model = keras.models.clone_model(model)
        self.model.add(Dense(units=real_code.m, activation='linear'))
        self.model.compile(loss=loss,
                           optimizer='adam',
                           metrics=['mse'])

    def fit(self, X_tr, y_tr, batch_size=None, epochs=12):
        Y_tr = np.array([self.real_code.encode(i) for i in y_tr.astype(int)])
        self.model.fit(X_tr, Y_tr, batch_size=batch_size, epochs=epochs, verbose=0)

    def predict(self, X_te):
        pts_pred = self.model.predict(X_te)
        y_pred = np.array([self.real_code.decode(pt_pred) for pt_pred in pts_pred])
        return pts_pred, y_pred

    def evaluate(self, X_te, y_te, plot=False):
        pts_pred, y_pred = self.predict(X_te)
        acc = len([y_i for i, y_i in enumerate(y_pred) if y_i == y_te[i]]) / len(y_pred)
        ys_most_frequent = set(filter(lambda y: np.count_nonzero(y_te == y) > 100, y_te))
        if plot:
            d = 0.1
            dx = max(np.abs(pts_pred[:, 0].min()), np.abs(pts_pred[:, 0].max())) + d
            dy = max(np.abs(pts_pred[:, 1].min()), np.abs(pts_pred[:, 1].max())) + d
            plt.axis((-dx, dx, -dy, dy))
            plt.scatter(pts_pred[:, 0], pts_pred[:, 1], color='red')  # plot predicted points
            # plt.scatter(self.real_code.pts[:, 0], self.real_code.pts[:, 1], color='blue') # plot class points
            pts_te = np.array([self.real_code.encode(y) for y in set(y_te) if y not in ys_most_frequent])
            pts_te_freq = np.array([self.real_code.encode(y) for y in ys_most_frequent])
            plt.scatter(pts_te[:, 0], pts_te[:, 1], color='blue') # plot test points
            plt.scatter(pts_te_freq[:, 0], pts_te_freq[:, 1], color='green') # plot test points
            plt.legend(['predicted', 'labels w/ <= 100 occurences', 'labels w > 100 occurences'])
            # x0s = np.linspace(-dx, dx, 2)  # plot hyper planes
            # for w in self.real_code.ws:
            #     x1s = [(-w[0] * x0 + 1) / w[1] for x0 in x0s]
            #     plt.plot(x0s, x1s)
            plt.show()
        return acc



