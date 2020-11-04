import numpy as np
from scipy.special import expit


# sigmoid function
def sigmoid():
    return lambda X: 1 / (1 + np.exp(-X))


# sigmoid function derivative
def sigmoid_der():
    return lambda X: sigmoid()(X) * (1 - sigmoid()(X))


# relu function
def relu():
    return lambda X: np.where(X >= 0, X, 0)


# relu function derivative
def relu_der():
    def _(X):
        X[X <= 0] = 0
        X[X > 0] = 1
        return X

    return _


# softmax function
def softmax():
    def _(X):
        exps = np.exp(X)
        summ = np.sum(X, axis=0)
        return np.divide(exps, summ)

    return _


# softmax derivative
# <CHECK-LATER>
# def softmax_der():
#     pass

def no_func():
    return lambda X: X


def no_func_der():
    return lambda X: 1


def get_activation(activation):
    activation = activation.lower()
    if activation == 'sigmoid':
        return sigmoid(), sigmoid_der()
    elif activation == 'relu':
        return relu(), relu_der()
    elif activation == 'no_func':
        return no_func(), no_func_der()
    # default
    return no_func(), no_func_der()


"""
    Layer : the hidden layer of neural network
    ....

    Attributes
    ----------
        shape: type -> INT
            is the number of neurons in this layer
        activation: type -> STRING
            the activation function of this layer
            default -> 'sigmoid'
"""


class Layer:
    def __init__(self, shape, activation='sigmoid'):
        self._act_function, self._act_function_der = get_activation(activation)
        self.shape = (shape,)

    # setup the hidden layer
    # config shape, weights, biases & initialize them
    def _setup(self, prev_layer):
        self.shape = (prev_layer.shape[0],) + self.shape
        self.weight = np.random.randn(prev_layer.shape[1], self.shape[1]) / self._get_spec_number(prev_layer)
        self.bias = np.random.randn(1, self.shape[1]) / self._get_spec_number(prev_layer)
        self.values = np.zeros(self.shape)

    def _get_spec_number(self, prev_layer):
        return self.shape[1] * prev_layer.shape[1]

    def _foward(self, prev_layer):
        if isinstance(prev_layer, np.ndarray):  # first hidden layer
            self.z = np.dot(prev_layer, self.weight)  # + self.bias
        else:
            self.z = np.dot(prev_layer.values, self.weight)  # + self.bias
        self.values = self._act_function(self.z)

    def _backward(self, delta, prev_layer, learning_rate):

        delta = delta * self._act_function_der(self.z)
        # NOT SURE ABOUT THE DERIVATIVE OF BIAS
        # <CHECK-LATER>
        delta_bias = np.sum(delta, axis=0).reshape(1, -1)
        if isinstance(prev_layer, np.ndarray):  # first hidden layer
            weight_der = np.dot(prev_layer.T, delta)
            # print(prev_layer.shape)
        else:
            weight_der = np.dot(prev_layer.values.T, delta)
        self.bias += learning_rate * delta_bias
        delta = np.dot(delta, self.weight.T)
        self.weight += learning_rate * weight_der
        return delta


"""
    NN: is a simple neural network model for classification & regression problems
    ....

    Attributes
    ----------
        X:  type -> np.ndarray
            the input data
        Y: type -> np.ndarray
            the target data
        output_activation: type-> string
            the activation function of the last layer,
            the output layer
            default -> 'sigmoid'

    Example
    -------
    > from NN import nn
    > from Layer import Layer
    > # import some data from sklearn library
    > from sklearn.datasets import load_breast_cancer
    > inputs = data.data
    > targets = data.target.reshape(-1,1)
    > neural_network_model = nn(inputs, targets)
    > # add hidden layers
    > neural_network_model.add_layer( Layer(32, activation='relu') )
    > neural_network_model.fit()
    > # predict data
    > Y_pred = neural_network_model.predict(INPUTS)
    > # plot cost function
    > import matplotlib.pyplot as plt
    > plt.plot(neural_network_model._costs)
    > plt.show()

"""


class NN:

    def __init__(self, X, Y, output_activation='sigmoid'):
        self._X = X
        self._Y = Y
        self._layers = []
        self._output_activation = output_activation
        self._m = self._X.shape[0]

    def add_layer(self, layer):
        if not isinstance(layer, Layer):
            raise Exception("Invalid Type", type(layer), " != <class 'Layer'>")
        self._layers.append(layer)

    # Train data
    def fit(self, learning_rate=0.1, iteration=1000):
        self._setup()
        self._costs = []
        self._learning_rate = learning_rate
        self._iteration = iteration
        for i in range(iteration):
            self._fowardPropagation()
            self._backPropagation()
            print(self._calc_cost(self._layers[len(self._layers) - 1].values))
            if (i % 100 == 0):
                self._costs.append(self._calc_cost(self._layers[len(self._layers) - 1].values))

    # return the cost function
    def _calc_cost(self, Y_pred):
        return np.sum(np.square(self._Y - Y_pred) / 2)

    # configuration the shape,
    # weight and bias of each layer
    # add output layer
    def _setup(self):
        for index, layer in enumerate(self._layers):
            if (index == 0):  # first hidden layer
                layer._setup(self._X)
            else:
                layer._setup(self._layers[index - 1])
        ### setup and add output layer
        output_layer = Layer(self._Y.shape[1], activation=self._output_activation)
        output_layer._setup(self._layers[len(self._layers) - 1])
        self.add_layer(output_layer)

    def _fowardPropagation(self):
        for index, layer in enumerate(self._layers):
            if (index == 0):  # first hidden layer
                layer._foward(self._X)
            else:
                layer._foward(self._layers[index - 1])

    def _backPropagation(self):
        delta = self._Y - self._layers[len(self._layers) - 1].values
        for i in range(len(self._layers) - 1, -1, -1):
            if (i == 0):  # first hidden layer
                delta = self._layers[i]._backward(delta, self._X, self._learning_rate/self._m)
            else:
                delta = self._layers[i]._backward(delta, self._layers[i - 1], self._learning_rate/self._m)

    def predict(self, X_test):
        for index, layer in enumerate(self._layers):
            if (index == 0):
                layer._foward(X_test)
            else:
                layer._foward(self._layers[index - 1])
        if self._is_continues():  # if target labels is continues
            return self._layers[len(self._layers) - 1].values
        if self._is_multiclass():  # if target labels is multiclass
            return self._threshold_multiclass(self._layers[len(self._layers) - 1])
        return self._threshold(self._layers[len(self._layers) - 1], 0.5)  # binary classification

    # set the 'predict.value' > 'value' [treshhold] to '1' others to '0'
    def _threshold(self, target, value):
        predict = target.values
        predict[predict < value] = 0
        predict[predict >= value] = 1
        return predict

    # set the max 'predict.value' to '1' others to '0'
    def _threshold_multiclass(self, target):
        predict = target.values
        predict = np.where(predict == np.max(predict, keepdims=True, axis=1), 1, 0)
        # predict[] = 1 | 0
        return predict

    # check if it's a multiclassfication problem
    def _is_multiclass(self):
        return len(np.unique(self._Y)) > 2

    # check if it's a regression problem
    def _is_continues(self):
        return len(np.unique(self._Y)) > (self._Y.shape[0] / 3)


