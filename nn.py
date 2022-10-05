import numpy as np
import matplotlib.pyplot as plt
import pickle
from numpy import indices

def display_image(image):
    """
    Displays an image from the mnist dataset

    Make sure you have the matplotlib library installed

    If using Jupyter, you may need to add %matplotlib inline to the top
    of your notebook
    """
    import matplotlib.pyplot as plt
    plt.imshow(image, cmap="gray")


def get_mnist_threes_nines(Y0=3, Y1=9):
    """
    Creates MNIST train / test datasets
    """
    import mnist

    y_train = mnist.train_labels()
    y_test = mnist.test_labels()
    X_train = (mnist.train_images()/255.0)
    X_test = (mnist.test_images()/255.0)
    train_idxs = np.logical_or(y_train == Y0, y_train == Y1)
    test_idxs = np.logical_or(y_test == Y0, y_test == Y1)
    y_train = y_train[train_idxs].astype('int')
    y_test = y_test[test_idxs].astype('int')
    X_train = X_train[train_idxs]
    X_test = X_test[test_idxs]
    y_train = (y_train == Y1).astype('int')
    y_test = (y_test == Y1).astype('int')
    return (X_train, y_train), (X_test, y_test)

def finite_difference_checker(f, x, k):
    """Returns \frac{\partial f}{\partial x_k}(x)"""
    e = 10**(-5)
    derivative = (f(x.copy().astype(float)[k]+e)-f(x.copy().astype(float)-e))/2*e
    return derivative

def sigmoid_activation(x):
    out = np.clip(np.where(x>=0,
        (1 / (1 + np.exp(-x))),
        ((np.exp(x))/ (1 + np.exp(x)))), 10**-15, 1 - 10**-15)
    return out, out * (1 - out)


def logistic_loss(g, y):
    assert (g.ndim == y.ndim)
    loss = -(y * np.log(g) + (1 - y) * np.log(1 - g))
    dL_dg = -((y / g) + ((y - 1)/(1 - g)))/len(g)
    return loss, dL_dg

def relu_activation(s):
    out = np.where(s > 0, s, 0)
    ds = np.where(s > 0, 1, 0)
    return out, ds

def layer_forward(x, W, b, activation_fn):
    out, grad = activation_fn(np.dot(x, W) + b)
    cache = (grad, x, W, b)
    return out, cache

def create_weight_matrices(layer_dims):
    """
    Creates a list of weight matrices defining the weights of NN
    
    Inputs:
    - layer_dims: A list whose size is the number of layers. layer_dims[i] defines
      the number of neurons in the i+1 layer.

    Returns a list of weight matrices
    """
    weights = [np.random.normal(0, 0.01, size=(e, layer_dims[i+1])) for i, e in enumerate(layer_dims) if i < len(layer_dims) - 1]
    return weights

def create_bias_vectors(layer_dims):
    biases = [np.random.normal(0, 0.01, size=(1, i)) for i in layer_dims[1:]]
    return biases

def forward_pass(X_batch, weight_matrices, biases, activations):
    layer_caches = []
    depth = len(weight_matrices)
    for i in range(depth):
        X_batch, layer_cache = layer_forward(X_batch, weight_matrices[i], biases[i], activations[i])
        layer_caches.append(layer_cache)
    return X_batch.flatten(), layer_caches

def backward_pass(dL_dg, layer_caches):

    depth = len(layer_caches)
    ### 0 = grad, 1 = x, 2 = W, 3 = b ###
    dW = np.multiply(dL_dg.reshape(len(dL_dg), -1), layer_caches[depth-1][0])
    grad_Ws = [np.dot(layer_caches[depth-1][1].T, dW)]
    #grad_Ws = [np.dot(layer_caches[depth-1][4].T, dW)]
    grad_bs = [np.sum(dW, axis=0)]

    for _, e in reversed(list(enumerate(layer_caches[:-1]))):
        dW = np.multiply(dW, e[0])
        grad_W = np.dot(e[1].T, dW)
        #grad_W = np.dot(e[4], dW)
        grad_Ws.insert(0, grad_W)
        grad_bs.insert(0, np.sum(dW, axis=0))
        dW = np.dot(dW, e[2].T).reshape(e[1].shape)

    return grad_Ws, grad_bs

from numpy import indices

(X_train, y_train), (X_test, y_test) = get_mnist_threes_nines()

'''
(12080, 28, 28)
(12080,)
(2019, 28, 28)
(2019,)
'''

bacth_size = 100
step_size = 0.1
epoches = 5
layer_dims = [784, 200, 1]
weight_matrices = create_weight_matrices(layer_dims)
bias_vectors = create_bias_vectors(layer_dims)
classifications = []
activations = [relu_activation, sigmoid_activation]
training_loss_list, training_accuracy_list, test_loss_list, test_accuracy_list = [], [], [], []

def change_weights(weight_matrices, bias_vectors, grad_W_matrix, grad_b_matrix):
    for el in range(len(weight_matrices)):
        weight_matrices[el] -= step_size * grad_W_matrix[el]
        bias_vectors[el] -= step_size * grad_b_matrix[el]
    return weight_matrices, bias_vectors

for _ in range(epoches):
    rows = X_train.shape[0]
    indices = np.arange(rows)
    np.random.shuffle(indices)
    for i in range(rows//bacth_size):
        X_batch = np.array([X_train[index] for index in indices[i*bacth_size:(i+1)*bacth_size]]).reshape(bacth_size, layer_dims[0])
        y_batch = np.array([y_train[index] for index in indices[i*bacth_size:(i+1)*bacth_size]])
        output, layer_caches = forward_pass(X_batch, weight_matrices, bias_vectors, activations)
        loss, dL_dg = logistic_loss(output, y_batch)
        grad_Ws, grad_bs = backward_pass(dL_dg, layer_caches)
        weight_matrices, bias_vectors = change_weights(weight_matrices, bias_vectors, grad_Ws, grad_bs)
        training_accuracy = sum(np.where(np.where(output >= 0.5, 1, 0)==y_batch, 1, 0))/len(y_batch)
        test_output, test_layer_caches = forward_pass(np.array(X_test).reshape(len(X_test), layer_dims[0]), weight_matrices, bias_vectors, activations)
        test_loss, _ = logistic_loss(test_output, y_test) 
        test_accuracy = sum(np.where(np.where(test_output >= 0.5, 1, 0)==y_test, 1, 0))/len(y_test)
        training_loss_list.append(np.mean(loss))
        training_accuracy_list.append(training_accuracy)
        test_loss_list.append(np.mean(test_loss))
        test_accuracy_list.append(test_accuracy)
        classifications.append((np.where(output < 0.5, 0, 1), i, indices))

plt.plot(training_loss_list)
plt.plot(test_loss_list)
plt.xlabel('Computation step')
plt.ylabel('Loss')
plt.show()

plt.plot(training_accuracy_list)
plt.plot(test_accuracy_list)
plt.xlabel('Computation step')
plt.ylabel('Accuracy')
plt.show()