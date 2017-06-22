from helper import load_data
import tensorflow as tf
import numpy as np

tf.python.control_flow_ops = tf

from keras.layers.core import Flatten, Dense
from keras.models import Sequential

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")
flags.DEFINE_string('epochs', int, "Number of times to run the train cycles")
flags.DEFINE_string('batch_size', int, "Size of train set to run network at a time")


def main(_):
    # load bottleneck data
    X_train, y_train = load_data(FLAGS.training_file)
    X_val, y_val = load_data(FLAGS.validation_file)

    print("train shape: ", X_train.shape, y_train.shape)
    print("val shape: ", X_val.shape, y_val.shape)

    nb_classes = len(np.unique(y_train))

    # define model
    input_shape = X_train.shape[1:]  # all except first
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(nb_classes, activation='softmax', input_shape=input_shape))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
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
