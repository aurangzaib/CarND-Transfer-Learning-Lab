import tensorflow as tf
import numpy as np
import pickle
import os

tf.python.control_flow_ops = tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from keras.layers.core import Flatten, Dense
from keras.layers import Input
from keras.models import Model

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('epochs', int, "Number of times to run the train cycles")
flags.DEFINE_string('batch_size', int, "Size of train set to run network at a time")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    x_train = train_data['features']
    y_train = train_data['labels']
    x_val = validation_data['features']
    y_val = validation_data['labels']

    return x_train, y_train, x_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)
    print("train shape: ", X_train.shape, y_train.shape)
    print("val shape: ", X_val.shape, y_val.shape)
    nb_classes = len(np.unique(y_train))
    print("classes : {}".format(nb_classes))
    # define model
    input_shape = X_train.shape[1:]  # all except first
    inp = Input(shape=input_shape)
    x = Flatten()(inp)
    x = Dense(nb_classes, activation='softmax')(x)
    model = Model(inp, x)
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    # train model
    model.fit(X_train, y_train,
              epochs=int(FLAGS.epochs),
              batch_size=int(FLAGS.batch_size),
              validation_data=(X_val, y_val),
              shuffle=True)


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()

'''

Run following commands from terminal:

1- To use VGG on cifar-10 dataset:
python feature_extraction.py --training_file vgg-100/vgg_cifar10_100_bottleneck_features_train.p --validation_file vgg-100/vgg_cifar10_bottleneck_features_validation.p --epochs 50 --batch_size 128

2- To use VGG on traffic dataset:
python feature_extraction.py --training_file vgg-100/vgg_traffic_100_bottleneck_features_train.p --validation_file vgg-100/vgg_traffic_bottleneck_features_validation.p --epochs 50 --batch_size 128

3- To use Inception(GoogLeNet) on cifar-10 dataset:
python feature_extraction.py --training_file inception-100/inception_cifar10_100_bottleneck_features_train.p --validation_file inception-100/inception_cifar10_bottleneck_features_validation.p --epochs 50 --batch_size 128

4- To use Inception(GoogLeNet) on traffic dataset:
python feature_extraction.py --training_file inception-100/inception_traffic_100_bottleneck_features_train.p --validation_file inception-100/inception_traffic_bottleneck_features_validation.p --epochs 50 --batch_size 128

5- To use ResNet on cifar-10 dataset:
python feature_extraction.py --training_file resnet-100/resnet_cifar10_100_bottleneck_features_train.p --validation_file resnet-100/resnet_cifar10_bottleneck_features_validation.p --epochs 50 --batch_size 128

6- To use ResNet on traffic dataset:
python feature_extraction.py --training_file resnet-100/resnet_traffic_100_bottleneck_features_train.p --validation_file resnet-100/resnet_traffic_bottleneck_features_validation.p --epochs 50 --batch_size 128

'''
