# O input da próxima camada é o output da camada anterior
import numpy as np
inputs = [[1.0,  2.0,  3.0, 2.5],
          [2.0,  5.0, -1.0, 2.0],
          [-1.5, 2.7,  3.3, -0.8]]

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
outputs = np.dot(inputs, np.array(weights).T) + biases
weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]
outputs2 = np.dot(outputs, np.array(weights2).T) + biases2

print("Inputs: ")
print(np.array(inputs))
print("Weights 1: ")
print(np.array(weights).T)
print("Outputs 1: ")
print(np.array(outputs))
print("Weights 2: ")
print(np.array(weights2).T)
print("Outputs 2: ")
print(outputs2)
