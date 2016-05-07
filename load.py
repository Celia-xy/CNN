
import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
import cv2
from pandas import DataFrame


FTRAIN = './data/training.csv'
FTEST = './data/test.csv'


def load(test=False, cols=None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're only interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname))  # load pandas dataframe

    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))

    if cols:  # get a subset of columns
        df = df[list(cols) + ['Image']]

    # print(df.count())  # prints the number of values for each column
    df = df.dropna()  # drop all rows that have missing values in them


    X_0 = np.vstack(df['Image'].values)
    X = np.zeros(X_0.shape)
    for i in range(X_0.shape[0]):
        im = X_0[i]
        im = im.reshape(96, 96)
        im = im.astype(np.uint8)
        eq = cv2.equalizeHist(im)
        eq = eq.reshape(1, 96*96)
        X[i] = eq
    X = X/255.  # scale pixel values to [0, 1]
    X = X.astype(np.float32)

    if not test:  # only FTRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y - 48) / 48  # scale target coordinates to [-1, 1]
        X, y = shuffle(X, y, random_state=42)  # shuffle train data
        y = y.astype(np.float32)
    else:
        y = None

    return X, y

def load2d(test=False, cols=None):
    X, y = load(test=test, cols=cols)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

def load_spe_train(win, test=False, cols=None):
    X, y = load(test, cols)
    x_co = np.zeros(y.shape[0])
    y_co = np.zeros(y.shape[0])

    X_new = np.zeros((X.shape[0], win*win))
    Y_new = np.zeros(y.shape)
    for i in range(0, y.shape[0]):
        x_co[i] = np.mean(y[i, 0:y.shape[1]/2])*48+48
        y_co[i] = np.mean(y[i, y.shape[1]/2:y.shape[1]])*48+48
        X_trans = np.array(X[i].reshape(96, 96))
        row = np.floor(y_co[i]-win/2)
        col = np.floor(x_co[i]-win/2)
        if row < 0:
            row = 0
            y_co[i] = y_co[i]+np.abs(row)
        if row+win > 96:
            row = 96-win
            y_co[i] = y_co[i]-(row+win-96)
        if col < 0:
            col = 0
            x_co[i] = x_co[i]+np.abs(col)
        if col+win > 96:
            col = 96-win
            x_co[i] = x_co[i]-(col+win-96)
        X_spe = X_trans[row:row+win, col:col+win]
        X_new[i] = X_spe.reshape(1, win*win)
        Y_new[i, 0:y.shape[1]/2] = y[i, 0:y.shape[1]/2]*48+48-x_co[i]
        Y_new[i, y.shape[1]/2:y.shape[1]] = y[i, y.shape[1]/2:y.shape[1]]*48+48-y_co[i]
    X_specialist = X_new.reshape(-1, 1, win, win)
    Y_specialist = 2*Y_new/win
    Y_specialist = Y_specialist.astype(np.float32)
    X_specialist = X_specialist.astype(np.float32)
    return X_specialist, Y_specialist

def load_spe_test(win, X_test, Y_pred, indices=None):

    X = X_test
    y = np.zeros((Y_pred.shape[0], indices.shape[0]))
    for i in range(0, Y_pred.shape[0]):
        for j in range(0, indices.shape[0]):
            y[i, j] = Y_pred[i, indices[j]-1]
    x_co = np.zeros(y.shape[0])
    y_co = np.zeros(y.shape[0])

    X_new = np.zeros((X.shape[0], win*win))
    Y_new = np.zeros(y.shape)
    for i in range(0, y.shape[0]):
        x_co[i] = np.mean(y[i, 0:y.shape[1]/2])*48+48
        y_co[i] = np.mean(y[i, y.shape[1]/2:y.shape[1]])*48+48
        X_trans = np.array(X[i].reshape(96, 96))
        row = np.floor(y_co[i]-win/2)
        col = np.floor(x_co[i]-win/2)
        if row < 0:
            row = 0
            y_co[i] = y_co[i]+np.abs(row)
        if row+win > 96:
            row = 96-win
            y_co[i] = y_co[i]-(row+win-96)
        if col < 0:
            col = 0
            x_co[i] = x_co[i]+np.abs(col)
        if col+win > 96:
            col = 96-win
            x_co[i] = x_co[i]-(col+win-96)
        X_spe = X_trans[row:row+win, col:col+win]
        X_new[i] = X_spe.reshape(1, win*win)
        Y_new[i, 0:y.shape[1]/2] = y[i, 0:y.shape[1]/2]*48+48-x_co[i]
        Y_new[i, y.shape[1]/2:y.shape[1]] = y[i, y.shape[1]/2:y.shape[1]]*48+48-y_co[i]
    X_specialist = X_new.reshape(-1, 1, win, win)
    Y_specialist = 2*Y_new/win
    Y_specialist = Y_specialist.astype(np.float32)
    X_specialist = X_specialist.astype(np.float32)
    return X_specialist, x_co, y_co

# X, y = load()
# import cPickle as pickle
# with open('images.pickle', 'wb') as f:
#    pickle.dump(X , f, -1)
#    f.close()
# with open('features.pickle', 'wb') as f:
#    pickle.dump(y , f, -1)
#    f.close()

# print X[1]
# cv2.imshow('im', X[1].reshape(96, 96))
# cv2.waitKey(0)
# cv2.destroyAllWindows()


