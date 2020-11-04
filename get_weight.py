import pandas as pd
import numpy as np
from numpy import genfromtxt
import cv2

path = 'resource/weight_and_bias_task7/'


def get_number(i):
    bias = genfromtxt(path + 'bias' + str(i) + '.csv', delimiter=',')
    weight = genfromtxt(path + 'weight' + str(i) + '.csv', delimiter=',')
    return (weight, bias)


(w0, b0) = get_number(0)
(w1, b1) = get_number(1)
(w2, b2) = get_number(2)


# sigmoid function
def sigmoid(X):
    return 1 / (1 + np.exp(-X))


# relu function
def relu(X):
    return  np.where(X >= 0, X, 0)


def predict(img):
    """
    :type x: image input
    """
    X = np.reshape(img, newshape=(1,-1))
    z0 = np.dot(X, w0) + b0
    value0 = relu(z0)
    z1 = np.dot(z0,w1) + b1
    value1 = sigmoid(z1)
    z2 = np.dot(value1,w2)
    predict = sigmoid(z2)
    if z2 > 0.5:
        return 1
    else :
        return 0


