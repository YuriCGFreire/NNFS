import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

output = []
output2 = []

# Fazendo sem a função max do python
for i in inputs:
    if(i > 0):
        output.append(i)
    else:
        output.append(0)

# Usando a função max do python
for i in inputs:
    output2.append(max(0, i))

# Usando numpy
output3 = np.maximum(0, inputs)

# print("Output 1: ", output)
# print("Output 2: ", output2)
# print("Output 3: ", output3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases

class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

        
X, y = spiral_data(samples=100, classes=3)
# print(X[:5])

layer_dense = Layer_Dense(2, 3)
relu = Activation_ReLU()
layer_dense.forward(X) 
relu.forward(layer_dense.output)
print(relu.output[:5])
