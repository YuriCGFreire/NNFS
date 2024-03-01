import nnfs
from nnfs.datasets import spiral_data
import numpy as np
import matplotlib.pyplot as plt

nnfs.init()
 
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        #Initialize weights and biases
        # O tamanho das dimensões da nossa matriz de pesos será igual a (n_inputs, n_neurons)
        # onde a quantidade de linha é igual a quantidade de inputs e a quantidade de colunas é igual a quantidade 
        # de neuronios
        # Os valores dos pesos serão iniciados de forma aleatoria 
        # Ao invés de fazer a transposição quando for fazer o forward eu ja inicializo os pesos para ficar da forma
        # transposta
        # Forma transposta (n_inputs, n_neurons)
        # Não transposta (n_neurons, n_inputs)
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        #Biases é um vetor de 1 linha e n colunas, sendo n a quantidade de neuronios na minha camada atual
        #E serão iniciados com 0
        self.biases = np.zeros((1, n_neurons))
    # Forward pass
    def forward(self, inputs):
        # Método responsável por fazer a soma dos produtos dos inputs com os pesos e biases
        self.output = np.dot(inputs, self.weights) + self.biases

# Camada exemplo com dados inventados
# inputs = np.array([1, 2, 3, 4]) # (1, 4)
# layer_dense = Layer_Dense(len(inputs), 3) #(4, 3)
# layer_dense.forward(inputs=inputs)
# print(layer_dense.output) #(1, 3)

# Produto de matrizes 
# Para que seja possível o tamanho da segunda dimensão da primeira matriz, tem que ser igual ao tamanho da primeira
# dimensão da segunda matriz
# A matriz resultante vai ter dimensões onde a primeira dimensão é igual a primeira dimensão da primeira matriz, 
# a segunda dimensão é igual a segunda dimensão da segunda matriz
        
X, y = spiral_data(samples=100, classes=3)

#Cada um dos nossos samples consiste em um vetor com dois valores, sendo coordenadas X e Y
#Então nossa camada de neuronios terão pesos (2, 3), sendo 3 a quantidade de neuronios

layer_dense = Layer_Dense(2, 3)
# print(layer_dense.weights)
layer_dense.forward(X) 
print(layer_dense.output[:5])