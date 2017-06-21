import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import os

tf.python.control_flow_ops = tf
from keras.layers.core import Activation, Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPool2D
from keras.models import Sequential

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def load_data(filename):
    import pickle
    import os
    root = os.getcwd() + '/small_traffic_set/'
    with open(root + filename, mode='rb') as f:
        data = pickle.load(f)
    assert (len(data['features']) == len(data['labels']))
    return data['features'], data['labels']


def grayscale(x):
    import cv2 as cv
    import numpy as np
    for index, image in enumerate(x):
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        im2 = np.zeros_like(image)
        im2[:, :, 0], im2[:, :, 1], im2[:, :, 2] = gray, gray, gray
        x[index] = im2
    return x


def normalizer(x):
    import numpy as np
    x_min = float(np.min(x))
    x_max = float(np.max(x))
    x = (x - x_min) / (x_max - x_min)
    return x


def pre_process(features, labels):
    assert (len(features) == len(labels))
    features = grayscale(features)
    features = normalizer(features)
    return features, labels


x_train, y_train = load_data('transforms-train.p')
x_test, y_test = load_data('test.p')

model = Sequential()
# Layer 1 -- convolution
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid', input_shape=(32, 32, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Activation(activation='relu'))
# Layer 2 -- convolution
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
model.add(Activation(activation='relu'))
# Layer 3 -- convolution
model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='valid'))
model.add(Activation(activation='relu'))
# flatten
model.add(Flatten(input_shape=(32, 32, 3)))
# Layer 4 -- fully connected layer
model.add(Dense(units=120))
model.add(Dropout(rate=0.5))
model.add(Activation(activation='relu'))
# Layer 5 -- fully connected layer
model.add(Dense(units=84))
model.add(Activation(activation='relu'))
model.add(Dropout(rate=0.5))
# Layer 6 -- output layer
model.add(Dense(units=43))
model.add(Activation(activation='softmax'))

x_train, y_train = pre_process(x_train, y_train)
binarizer = LabelBinarizer()
y_train_one_hot = binarizer.fit_transform(y_train)

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(x=x_train, y=y_train_one_hot, batch_size=128, epochs=3, validation_split=0.3, shuffle=True, verbose=0)

x_test, y_test = pre_process(x_test, y_test)
binarizer = LabelBinarizer()
y_test_one_hot = binarizer.fit_transform(y_test)

metrics = model.evaluate(x=x_test, y=y_test_one_hot, verbose=0)
for metrics_index, metrics_name in enumerate(model.metrics_names):
    name = metrics_name
    value = metrics[metrics_index]
    print("{}: {}%".format(name, value * 100))
