import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x):
    return np.where(x > 0, x, 0.01 * x)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

activations = {
    "ReLU": relu,
    "LeakyReLU": leaky_relu,
    "Sigmoid": sigmoid,
    "Tanh": tanh
}

plt.figure()
for name, func in activations.items():
    plt.plot(x, func(x), label=name)

plt.legend()
plt.title("Comparaison")
plt.grid()
plt.show()