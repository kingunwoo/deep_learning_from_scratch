import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    sum = np.sum(np.exp(x))
    return np.exp(x)/sum

def tanh(x):
    return

def ReLU(x):
    return np.maximum(0,x)

