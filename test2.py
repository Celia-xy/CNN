__author__ = 'Alex'
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
import cPickle as pickle
import cv2

# window for subtract specific area
win = 38

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

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        for i in range(0, np.int64(bs/2)):
            X_trans = Xb[indices[i]].reshape(96, 96)
            X_trans = X_trans[:, ::-1]
            Xb[indices[i]] = X_trans.reshape(1, 9216)

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

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
win = 38


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



with open('./data/net4.pickle', 'rb') as f2:
    net4 = pickle.load(f2)

with open('./data/SpeLE.pickle', 'rb') as f2:
    netLE = pickle.load(f2)

with open('./data/SpeRE.pickle', 'rb') as f2:
    netRE = pickle.load(f2)

with open('./data/SpeN.pickle', 'rb') as f2:
    netN = pickle.load(f2)

with open('./data/SpeM.pickle', 'rb') as f2:
    netM = pickle.load(f2)


train_lossLE = np.array([i["train_loss"] for i in netLE.train_history_])
valid_lossLE = np.array([i["valid_loss"] for i in netLE.train_history_])
train_lossRE = np.array([i["train_loss"] for i in netRE.train_history_])
valid_lossRE = np.array([i["valid_loss"] for i in netRE.train_history_])
train_lossN = np.array([i["train_loss"] for i in netN.train_history_])
valid_lossN = np.array([i["valid_loss"] for i in netN.train_history_])
train_lossM = np.array([i["train_loss"] for i in netM.train_history_])
valid_lossM = np.array([i["valid_loss"] for i in netM.train_history_])


train_loss4 = np.array([i["train_loss"] for i in net4.train_history_])
valid_loss4 = np.array([i["valid_loss"] for i in net4.train_history_])

pyplot.plot(train_loss4[0:999], linewidth=3, label="Full face train loss", color='b', linestyle='--')
pyplot.plot(valid_loss4[0:999], linewidth=3, label="Full face valid loss", color='b')
pyplot.plot(train_lossLE, linewidth=3, label="NetLE_21 train loss", color='g', linestyle='--')
pyplot.plot(valid_lossLE, linewidth=3, label="NetLE_21 valid loss", color='g')
pyplot.plot(train_lossRE, linewidth=3, label="NetRE_21 train loss", color='r', linestyle='--')
pyplot.plot(valid_lossRE, linewidth=3, label="NetRE_21 valid loss", color='r')
pyplot.plot(train_lossN, linewidth=3, label="NetN_21 train loss", color='c', linestyle='--')
pyplot.plot(valid_lossN, linewidth=3, label="NetN_21 valid loss", color='c')
pyplot.plot(train_lossM, linewidth=3, label="NetM_21 train loss", color='m', linestyle='--')
pyplot.plot(valid_lossM, linewidth=3, label="NetM_21 valid loss", color='m')

# pyplot.text(650, 0.0035, 'Final net4 train loss'+str(train_loss4[-1]), color='orange')
# pyplot.text(650, 0.0040, 'Final net4 valid loss'+str(valid_loss4[-1]), color='b')
pyplot.grid()
pyplot.legend()
pyplot.ylabel(r"$\varepsilon$", size=20)
pyplot.ylim(1e-3, 1e-2)
pyplot.yscale("log")
pyplot.show()


########################################################
#  load the full net and the specialists sequentially  #
########################################################


