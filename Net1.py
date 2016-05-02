
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
import load
import matplotlib.pyplot as pyplot
import numpy as np
from pandas import DataFrame
from pandas.io.parsers import read_csv
from datetime import datetime
import os


net1 = NeuralNet(
    layers=[  # three layers: one hidden layer
        ('input', layers.InputLayer),
        ('hidden', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    # layer parameters:
    input_shape=(None, 9216),  # 96x96 input pixels per batch
    hidden_num_units=100,  # number of units in hidden layer
    output_nonlinearity=None,  # output layer uses identity function
    output_num_units=30,  # 30 target values

    # optimization method:
    update=nesterov_momentum,
    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=True,  # flag to indicate we're dealing with regression problem
    max_epochs=1000,  # we want to train this many epochs
    verbose=1,
    )

X, y = load.load()
net1.fit(X, y)

# plot random 16 images
# def plot_sample(x, y, axis):
#     img = x.reshape(96, 96)
#     axis.imshow(img, cmap='gray')
#     axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)

# X, _ = load.load(test=True)
# y_pred = net1.predict(X)

# fig = pyplot.figure(figsize=(6, 6))
# fig.subplots_adjust(
#     left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)

# for i in range(8):
#     ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
#     plot_sample(X[i+100], y_pred[i+100], ax)
# pyplot.show()


print 'start making pickle'
# pickle the trained net
import cPickle as pickle
with open('net1.pickle', 'wb') as f:
    pickle.dump(net1, f, -1)
    f.close()

print 'start making submission'
# make submission
def submission(fname_net='net1.pickle'):
    with open(fname_net, 'rb') as f:
        net = pickle.load(f)     # net = specialists

    X = load.load(test=True)[0]
    y_pred = net.predict(X)
    print 'Finish predicting test file'
    columns = 'left_eye_center_x,left_eye_center_y,right_eye_center_x,right_eye_center_y,left_eye_inner_corner_x,left_eye_inner_corner_y,left_eye_outer_corner_x,left_eye_outer_corner_y,right_eye_inner_corner_x,right_eye_inner_corner_y,right_eye_outer_corner_x,right_eye_outer_corner_y,left_eyebrow_inner_end_x,left_eyebrow_inner_end_y,left_eyebrow_outer_end_x,left_eyebrow_outer_end_y,right_eyebrow_inner_end_x,right_eyebrow_inner_end_y,right_eyebrow_outer_end_x,right_eyebrow_outer_end_y,nose_tip_x,nose_tip_y,mouth_left_corner_x,mouth_left_corner_y,mouth_right_corner_x,mouth_right_corner_y,mouth_center_top_lip_x,mouth_center_top_lip_y,mouth_center_bottom_lip_x,mouth_center_bottom_lip_y'
    columns = columns.split(',')

    y_pred = y_pred * 48 + 48
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
submission()

# plot the valid and train loss
