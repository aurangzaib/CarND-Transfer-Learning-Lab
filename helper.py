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
