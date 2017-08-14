import numpy as np
import matplotlib.pylab as plt

def stepFunc(x):
    y = x > 0
    return y.astype(np.int)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def draw():
    x = np.arange(-5.0, 5.0, 0.1)
    y1 = stepFunc(x)
    y2 = sigmoid(x)
    y3 = relu(x)
    plt.plot(x, y1, label = "step")
    plt.plot(x, y2, linestyle = "--", label = "sigmoid")
    plt.plot(x, y3, linestyle = "-.", label = "ReLU")
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.show()