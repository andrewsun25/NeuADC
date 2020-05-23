import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.layers import Dense


def euc_dist_loss(y_true, y_pred):
    """
    Computes the sum of Euclidean distances between true and predicted points.
    :param y_true: List of true points.
    :param y_pred: List of predicted points.
    :return: Euclidean loss between y_true and y_pred.
    """
    import keras.backend as K
    return K.sqrt(K.sum(K.square(y_true - y_pred), axis=-1, keepdims=True))


def simple_model(input_shape):
    """
    :param input_shape: Shape of the first layer.
    :return: Simple 1 layer Keras NN.
    """
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dropout(0.2),
    ])
    return model


class NeuADC:
    def __init__(self, real_code, model, loss=euc_dist_loss):
        """
        A NN wrapper around a Keras NN.
        :param real_code: Encoding/decoding scheme to use.
        :param model: Keras NN model without a final output layer.
        :param loss: Loss function to train NN with.
        """
        self.real_code = real_code
        self.model = keras.models.clone_model(model)
        self.model.add(Dense(units=real_code.m, activation='linear'))
        self.model.compile(loss=loss,
                           optimizer='adam',
                           metrics=['mse'])

    def fit(self, X_tr, y_tr, batch_size=None, epochs=12):
        """
        Train on the given data.
        :param X_tr: Training set.
        :param y_tr: Labels of training set.
        :param batch_size: Number examples to use per update.
        :param epochs: Number of epochs (full run throughs) to train NN.
        """
        Y_tr = np.array([self.real_code.encode(i) for i in y_tr.astype(int)])
        self.model.fit(X_tr, Y_tr, batch_size=batch_size, epochs=epochs, verbose=0)

    def predict(self, X_te):
        """
        Predict on test set.
        :param X_te: Test set.
        :return: Predicted class points, class labels.
        """
        pts_pred = self.model.predict(X_te)
        y_pred = np.array([self.real_code.decode(pt_pred) for pt_pred in pts_pred])
        return pts_pred, y_pred

    def evaluate(self, X_te, y_te, plot=False):
        """
        Computes the accuracy of model on test data.
        :param X_te: Test set.
        :param y_te: Test labels.
        :param plot: Whether or not to plot predicted vs true points.
        :return: Accuracy of model.
        """
        pts_pred, y_pred = self.predict(X_te)
        acc = len([y_i for i, y_i in enumerate(y_pred) if y_i == y_te[i]]) / len(y_pred)
        if plot:
            d = 0.1
            dx = max(np.abs(pts_pred[:, 0].min()), np.abs(pts_pred[:, 0].max())) + d
            dy = max(np.abs(pts_pred[:, 1].min()), np.abs(pts_pred[:, 1].max())) + d
            plt.axis((-dx, dx, -dy, dy))
            plt.scatter(pts_pred[:, 0], pts_pred[:, 1], color='red')  # plot predicted points
            pts_te = np.array([self.real_code.encode(y) for y in set(y_te)])
            plt.scatter(pts_te[:, 0], pts_te[:, 1], color='blue')  # plot test points
            plt.legend(['predicted class points', 'actual class points'])
            plt.title('N = {}'.format(self.real_code.N))
            plt.show()
        return acc
