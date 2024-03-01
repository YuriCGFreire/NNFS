import numpy as np
import nnfs
from nnfs.datasets import spiral_data
nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
    
class Activation_Softmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


X, y = spiral_data(samples=100, classes=3)

# Primeira camada interna, que recebe os inputs
layer_dense1 = Layer_Dense(2, 3)
activation_relu = Activation_ReLU()
layer_dense1.forward(X)
activation_relu.forward(layer_dense1.output)

# Camada de saida que recebe como input o output da primeira camada interna
layer_dense2 = Layer_Dense(3, 3)
activation_softmax = Activation_Softmax()
layer_dense2.forward(activation_relu.output)
activation_softmax.forward(layer_dense2.output)
print(activation_softmax.output[:5])