import matplotlib.pyplot as plt
import pandas as pd
from keras_lr_finder import LRFinder
from sklearn.model_selection import train_test_split

from NeuADC import NeuADC, simple_model
from RealCode import RealCode


def plot_lr(model, X_train, y_train):
    """
    Plots learning rate vs loss for model on training set.
    :param model: Keras model to find learning rate of.
    :param X_train: Training data.
    :param y_train: Training labels.
    """
    lr_finder = LRFinder(model)
    # Adam default LR = 0.001
    lr_finder.find(X_train, y_train, start_lr=0.0001, end_lr=1)
    lr_finder.plot_loss()
    plt.show()


def normalize(df):
    """
    Normalizes all numerical features to [0,1].
    :param df: Dataframe to normalize.
    :return: Normalized copy of df.
    """
    result = df.copy()
    for feature_name in df.select_dtypes(include='number'):
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def load_abalone(plot_y=False):
    """
    Loads the abalone dataset into training/testing sets with 25% going to test set.
    :param plot_y: If yes, plot histogram of class labels.
    :return: Loaded train/test sets.
    """
    fp = 'data/abalone.data'
    df = pd.read_csv(fp, sep=',', header=None)
    ys = df.pop(8)
    N = ys.max()
    ys -= 1
    df = normalize(df)
    df = pd.get_dummies(df, columns=[0])
    if plot_y:
        plt.hist(ys, bins=N)
        plt.xlabel('Age')
        plt.ylabel('Count')
        plt.show()
    ys_most_frequent = set(filter(lambda y: len(ys[ys == y]) > 300, ys))
    ys = ys.values
    X_train, X_test, y_train, y_test = train_test_split(df, ys, test_size=0.25)
    X_train = X_train.values.reshape(X_train.shape + (1,))
    X_test = X_test.values.reshape(X_test.shape + (1,))
    return X_train, X_test, y_train, y_test, N, ys_most_frequent


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, N, ys_most_frequent = load_abalone()
    for i in range(5):
        real_code = RealCode(N, 2)
        model = NeuADC(real_code, simple_model(X_train.shape[1:]))
        model.fit(X_train, y_train)
        acc = model.evaluate(X_test, y_test, plot=True)
        print('acc: ', acc)
