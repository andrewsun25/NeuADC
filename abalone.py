import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

from NeuADC import NeuADC, simple_model, regression_model
from RealCode import RealCode
from keras_lr_finder import LRFinder

def plot_lr(model, X_train, y_train):
    lr_finder = LRFinder(model)
    # Adam default LR = 0.001
    lr_finder.find(X_train, y_train, start_lr=0.0001, end_lr=1)
    lr_finder.plot_loss()
    plt.show()

def normalize(df):
    result = df.copy()
    for feature_name in df.select_dtypes(include='number'):
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        result[feature_name] = (df[feature_name] - min_value) / (max_value - min_value)
    return result


def load_abalone(plot_y=False):
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


X_train, X_test, y_train, y_test, N, ys_most_frequent = load_abalone()
# X_train = np.array([x for i, x in enumerate(X_train) if y_train[i] in ys_most_frequent])
# X_test = np.array([x for i, x in enumerate(X_test) if y_test[i] in ys_most_frequent])
# y_train = np.array([y for y in y_train if y in ys_most_frequent])
# y_test = np.array([y for y in y_test if y in ys_most_frequent])

for i in range(5):
    real_code = RealCode(N, 2)
    model = NeuADC(real_code, simple_model(X_train.shape[1:]))
    model.fit(X_train, y_train)
    acc = model.evaluate(X_test, y_test, plot=True)
    print('acc: ', acc)

# reg_model = regression_model(X_train.shape[1:])
# plot_lr(reg_model, X_train, y_train)
# # reg_model.fit(X_train, y_train, batch_size=32, epochs=50)
# # y_pred = reg_model.predict(X_test)
# # reg_model.evaluate(X_test, y_test)

# ms = range(2, 20, 2)
# accs = []

    # accs.append(acc)
# plt.plot(ms, accs)
# plt.show()
