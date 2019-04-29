# Imports
import numpy as np
from keras.utils import np_utils
from mnist import MNIST

# Defining functions to load EMNIST data

def load_data(path):

    # Read all EMNIST test and train data
    mndata = MNIST(path)
    X_train, y_train = mndata.load(path + '/emnist-balanced-train-images-idx3-ubyte',
                                   path + '/emnist-balanced-train-labels-idx1-ubyte')
    X_test, y_test = mndata.load(path + '/emnist-balanced-test-images-idx3-ubyte',
                                 path + '/emnist-balanced-test-labels-idx1-ubyte')

    # Read mapping of the labels and convert ASCII values to chars
    mapping = []

    with open(path + '/emnist-balanced-mapping.txt') as f:
        for line in f:
            mapping.append(chr(int(line.split()[1])))

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    X_train = normalize(X_train)
    X_test = normalize(X_test)

    X_train = reshape_for_cnn(X_train)
    X_test = reshape_for_cnn(X_test)

    y_train = preprocess_labels(y_train, len(mapping))
    y_test = preprocess_labels(y_test, len(mapping))

    return X_train, y_train, X_test, y_test, mapping



def normalize(array):
    array = array.astype('float32')
    array /= 255

    return array


def reshape_for_cnn(array, color_channels=1, img_width=28, img_height=28):
    return array.reshape(array.shape[0], color_channels, img_width, img_height)


def preprocess_labels(array, nb_classes):
    return np_utils.to_categorical(array, nb_classes)
