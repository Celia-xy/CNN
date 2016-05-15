
__author__ = 'Celia'
###################################################################################
# In this program, we train the 4 specialists -- left eye, right eye, nose, mouse #
# -- in cascade level 2, other specialists were all trained in the same way and   #
# saved as pickles in the data file.                                              #
###################################################################################

from lasagne import layers
from nolearn.lasagne import NeuralNet, BatchIterator
import load
import numpy as np
from datetime import datetime
from pandas import DataFrame
from pandas.io.parsers import read_csv
import os
import theano
from matplotlib import pyplot

# window for subtract specific area
win = 32

# early stopping for specialists
class EarlyStopping(object):
    def __init__(self, patience=100):
        self.patience = patience
        self.best_valid = np.inf
        self.best_valid_epoch = 0
        self.best_weights = None

    def __call__(self, nn, train_history):
        current_valid = train_history[-1]['valid_loss']
        current_epoch = train_history[-1]['epoch']
        if current_valid < self.best_valid:
            self.best_valid = current_valid
            self.best_valid_epoch = current_epoch
            self.best_weights = [w.get_value() for w in nn.get_all_params()]
        elif self.best_valid_epoch + self.patience < current_epoch:
            print("Early stopping.")
            print("Best valid loss was {:.6f} at epoch {}.".format(
                self.best_valid, self.best_valid_epoch))
            nn.load_weights_from(self.best_weights)
            raise StopIteration()

class FlipBatchIterator2d(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator2d, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

# Changing learning rate and momentum over time


def float32(k):
    return np.cast['float32'](k)


class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None

    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)

        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

net_spe_M = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, win, win),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2), dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2), dropout2_p=0.2,
    hidden3_num_units=200,
    output_num_units=8, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200)
        ],
    max_epochs=1000,
    verbose=1,
    )
net_spe_LE = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, win, win),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2), dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2), dropout2_p=0.2,
    hidden3_num_units=200,
    output_num_units=10, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200)
        ],
    max_epochs=1000,
    verbose=1,
    )
net_spe_RE = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, win, win),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2), dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2), dropout2_p=0.2,
    hidden3_num_units=200,
    output_num_units=10, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200)
        ],
    max_epochs=1000,
    verbose=1,
    )
net_spe_N = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, win, win),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2), dropout1_p=0.1,
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2), dropout2_p=0.2,
    hidden3_num_units=200,
    output_num_units=4, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        EarlyStopping(patience=200)
        ],
    max_epochs=1000,
    verbose=1,
    )


# parameters


columns_LE = ('left_eye_center_x', 'left_eye_inner_corner_x', 'left_eye_outer_corner_x', 'left_eyebrow_inner_end_x', 'left_eyebrow_outer_end_x',
            'left_eye_center_y', 'left_eye_inner_corner_y', 'left_eye_outer_corner_y', 'left_eyebrow_inner_end_y', 'left_eyebrow_outer_end_y',)
columns_RE = ('right_eye_center_x', 'right_eye_inner_corner_x', 'right_eye_outer_corner_x', 'right_eyebrow_inner_end_x', 'right_eyebrow_outer_end_x',
            'right_eye_center_y', 'right_eye_inner_corner_y', 'right_eye_outer_corner_y', 'right_eyebrow_inner_end_y', 'right_eyebrow_outer_end_y',)
columns_N = ('nose_tip_x', 'mouth_center_top_lip_x', 'nose_tip_y', 'mouth_center_top_lip_y',)
columns_M = ('mouth_left_corner_x', 'mouth_right_corner_x', 'mouth_center_top_lip_x', 'mouth_center_bottom_lip_x',
           'mouth_left_corner_y', 'mouth_right_corner_y', 'mouth_center_top_lip_y', 'mouth_center_bottom_lip_y',)

indices_LE = np.array([1, 5, 7, 13, 15, 2, 6, 8, 14, 16])
indices_RE = np.array([3, 9, 11, 17, 19, 4, 10, 12, 18, 20])
indices_N = np.array([21, 27, 22, 28])
indices_M = np.array([23, 25, 27, 29, 24, 26, 28, 30])

# fit the specialists and save them
# LE
X_LE, Y_LE = load.load_spe_train(win=win, cols=columns_LE)  # load 2-d data
net_spe_LE.fit(X_LE, Y_LE)
# pickle the trained specialists
import cPickle as pickle
with open('SpeLE2.pickle', 'wb') as f1:
    pickle.dump(net_spe_LE, f1, -1)


# RE
X_RE, Y_RE = load.load_spe_train(win=win, cols=columns_RE)  # load 2-d data
net_spe_RE.fit(X_RE, Y_RE)
# pickle the trained specialists
import cPickle as pickle
with open('SpeRE2.pickle', 'wb') as f2:
    pickle.dump(net_spe_RE, f2, -1)


# N
X_N, Y_N = load.load_spe_train(win=win, cols=columns_N)  # load 2-d data
net_spe_N.fit(X_N, Y_N)
# pickle the trained specialists
import cPickle as pickle
with open('SpeN2.pickle', 'wb') as f3:
    pickle.dump(net_spe_N, f3, -1)

# M
X_M, Y_M = load.load_spe_train(win=win, cols=columns_M)  # load 2-d data
net_spe_M.fit(X_M, Y_M)
# pickle the trained specialists
import cPickle as pickle
with open('SpeM2.pickle', 'wb') as f4:
    pickle.dump(net_spe_M, f4, -1)


########################################################
#  load the full net and the specialists sequentially  #
########################################################


# ------ make full image prediction first ----------- #

with open('net4.pickle', 'rb') as f0:
     net_full = pickle.load(f0)

X = load.load2d(test=True)[0]
print X.shape
y_pred = net_full.predict(X)
y_pred_final1 = np.zeros(y_pred.shape)         # array to save the final predictions of each specialists
y_pred_final2 = np.zeros(y_pred.shape)

# ------ load the trained nets together to make the final predict ---------- #

# LE
# with open('SpeLE.pickle', 'rb') as f2:
#      net_spe_LE = pickle.load(f2)

X_LE_test, x_co_LE, y_co_LE = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_LE)
y_LE_pred = net_spe_LE.predict(X_LE_test)
y_LE_pred_big = np.zeros(y_LE_pred.shape)
num_r = y_LE_pred.shape[0]
num_c = y_LE_pred.shape[1]
for i in range(0, num_r):
    y_LE_pred_big[i, 0:num_c/2] = y_LE_pred[i, 0:num_c/2]*win/2 + x_co_LE[i]
    y_LE_pred_big[i, num_c/2:num_c] = y_LE_pred[i, num_c/2:num_c]*win/2 + y_co_LE[i]
    for j in range(0, num_c):
        y_pred_final1[i, indices_LE[j]-1] = y_LE_pred_big[i, j]

# RE
# with open('SpeRE.pickle', 'rb') as f3:
#      net_spe_RE = pickle.load(f3)

X_RE_test, x_co_RE, y_co_RE = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_RE)
y_RE_pred = net_spe_RE.predict(X_RE_test)
y_RE_pred_big = np.zeros(y_RE_pred.shape)
num_r = y_RE_pred.shape[0]
num_c = y_RE_pred.shape[1]
for i in range(0, num_r):
    y_RE_pred_big[i, 0:num_c/2] = y_RE_pred[i, 0:num_c/2]*win/2 + x_co_RE[i]
    y_RE_pred_big[i, num_c/2:num_c] = y_RE_pred[i, num_c/2:num_c]*win/2 + y_co_RE[i]
    for j in range(0, num_c):
        y_pred_final1[i, indices_RE[j]-1] = y_RE_pred_big[i, j]

# N
# with open('SpeN.pickle', 'rb') as f4:
#      net_spe_N = pickle.load(f4)

X_N_test, x_co_N, y_co_N = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_N)
y_N_pred = net_spe_N.predict(X_N_test)
y_N_pred_big = np.zeros(y_N_pred.shape)
num_r = y_N_pred.shape[0]
num_c = y_N_pred.shape[1]
for i in range(0, num_r):
    y_N_pred_big[i, 0:num_c/2] = y_N_pred[i, 0:num_c/2]*win/2 + x_co_N[i]
    y_N_pred_big[i, num_c/2:num_c] = y_N_pred[i, num_c/2:num_c]*win/2 + y_co_N[i]
    y_pred_final1[i, 20] = y_N_pred_big[i, 0]
    y_pred_final1[i, 21] = y_N_pred_big[i, 2]

# M
# with open('SpeM.pickle', 'rb') as f1:
#      net_spe_M = pickle.load(f1)

X_M_test, x_co_M, y_co_M = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_M)
y_M_pred = net_spe_M.predict(X_M_test)
y_M_pred_big = np.zeros(y_M_pred.shape)
num_r = y_M_pred.shape[0]
num_c = y_M_pred.shape[1]
for i in range(0, num_r):
    y_M_pred_big[i, 0:num_c/2] = y_M_pred[i, 0:num_c/2]*win/2 + x_co_M[i]
    y_M_pred_big[i, num_c/2:num_c] = y_M_pred[i, num_c/2:num_c]*win/2 + y_co_M[i]
    for j in range(0, num_c):
        y_pred_final1[i, indices_M[j]-1] = y_M_pred_big[i, j]

# -------- load the win = 38 networks ------- #
win = 38
# LE
with open('SpeLE.pickle', 'rb') as f2:
      net_spe_LE = pickle.load(f2)

X_LE_test, x_co_LE, y_co_LE = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_LE)
y_LE_pred = net_spe_LE.predict(X_LE_test)
y_LE_pred_big = np.zeros(y_LE_pred.shape)
num_r = y_LE_pred.shape[0]
num_c = y_LE_pred.shape[1]
for i in range(0, num_r):
    y_LE_pred_big[i, 0:num_c/2] = y_LE_pred[i, 0:num_c/2]*win/2 + x_co_LE[i]
    y_LE_pred_big[i, num_c/2:num_c] = y_LE_pred[i, num_c/2:num_c]*win/2 + y_co_LE[i]
    for j in range(0, num_c):
        y_pred_final2[i, indices_LE[j]-1] = y_LE_pred_big[i, j]

# RE
with open('SpeRE.pickle', 'rb') as f3:
      net_spe_RE = pickle.load(f3)

X_RE_test, x_co_RE, y_co_RE = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_RE)
y_RE_pred = net_spe_RE.predict(X_RE_test)
y_RE_pred_big = np.zeros(y_RE_pred.shape)
num_r = y_RE_pred.shape[0]
num_c = y_RE_pred.shape[1]
for i in range(0, num_r):
    y_RE_pred_big[i, 0:num_c/2] = y_RE_pred[i, 0:num_c/2]*win/2 + x_co_RE[i]
    y_RE_pred_big[i, num_c/2:num_c] = y_RE_pred[i, num_c/2:num_c]*win/2 + y_co_RE[i]
    for j in range(0, num_c):
        y_pred_final2[i, indices_RE[j]-1] = y_RE_pred_big[i, j]

# N
with open('SpeN.pickle', 'rb') as f4:
      net_spe_N = pickle.load(f4)

X_N_test, x_co_N, y_co_N = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_N)
y_N_pred = net_spe_N.predict(X_N_test)
y_N_pred_big = np.zeros(y_N_pred.shape)
num_r = y_N_pred.shape[0]
num_c = y_N_pred.shape[1]
for i in range(0, num_r):
    y_N_pred_big[i, 0:num_c/2] = y_N_pred[i, 0:num_c/2]*win/2 + x_co_N[i]
    y_N_pred_big[i, num_c/2:num_c] = y_N_pred[i, num_c/2:num_c]*win/2 + y_co_N[i]
    y_pred_final2[i, 20] = y_N_pred_big[i, 0]
    y_pred_final2[i, 21] = y_N_pred_big[i, 2]

# M
with open('SpeM.pickle', 'rb') as f1:
      net_spe_M = pickle.load(f1)

X_M_test, x_co_M, y_co_M = load.load_spe_test(win=win, X_test=X, Y_pred=y_pred, indices=indices_M)
y_M_pred = net_spe_M.predict(X_M_test)
y_M_pred_big = np.zeros(y_M_pred.shape)
num_r = y_M_pred.shape[0]
num_c = y_M_pred.shape[1]
for i in range(0, num_r):
    y_M_pred_big[i, 0:num_c/2] = y_M_pred[i, 0:num_c/2]*win/2 + x_co_M[i]
    y_M_pred_big[i, num_c/2:num_c] = y_M_pred[i, num_c/2:num_c]*win/2 + y_co_M[i]
    for j in range(0, num_c):
        y_pred_final2[i, indices_M[j]-1] = y_M_pred_big[i, j]

# ----- Average the full image prediction and specialists predictions ------- #
y_pred = y_pred*48+48
y_average = np.add(y_pred_final1, y_pred)
y_average = np.add(y_average, y_pred_final2)/3


# ----- make submission --------- #
print 'start making submission'
def submission(y_in):

    print 'Finish predict test file'
    columns = 'left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y'
    columns = columns.split(',')

    y_pred = y_in
    y_pred = y_pred.clip(0, 96)
    df = DataFrame(y_pred, columns=columns)

    lookup_table = read_csv(os.path.expanduser('./data/IdLookupTable.csv'))
    values = []

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            df.ix[row.ImageId - 1][row.FeatureName],
            ))

    now_str = datetime.now().isoformat().replace(':', '-')
    submission = DataFrame(values, columns=('RowId', 'Location'))
    filename = 'submission-{}.csv'.format(now_str)
    submission.to_csv(filename, index=False)
    print("Wrote {}".format(filename))
submission(y_average)


# -------- plot random images ------- #
# - plot specialists results - #
def plot_sample(x, y, axis):
     img = x
     axis.imshow(img, cmap='gray')
     axis.scatter(y[0::2], y[1::2], marker='x', s=10)

# - plot average results - #
fig = pyplot.figure(figsize=(6, 6))
fig.subplots_adjust(
     left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

for i in range(0, 8):
     ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
     plot_sample(X[i+100, 0, :, :], y_average[i+100], ax)

pyplot.show()








#########################
# When doing prediction #
#########################

# with open('net4.pickle', 'rb') as f:
#    net_full = pickle.load(f)

# with open('net_spe_M.pickle', 'rb') as f:
#    net_M = pickle.load(f)


