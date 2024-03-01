import numpy as np

inputs = [1, 2, 3, 2.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5]

# Se eu passar o input primeiro ele vai dar erro de shape. Ele vai levar como base o tamanho do array de inputs
# para fazer a multiplicação dos inputs com os neuronios. O array de inputs tem 4 elementos, quando ele chegar no 
# quarto laço for ele nao vai encontrar um quarto neuronio no nosso array de array (weights) e causara o erro. 
# Então devemos pensar da seguinte forma para saber qual elemento passar primeiro 
# Nos temos tres neuronios, sera feito o dot product de cada um desses neuronios com o array de inputs, por tanto
# a saida deve ser um array de output igual ao numero de neuronios da nossa camada
# Nesse caso o array de output deve conter 3 elementos, pois temos 3 neuronios

layer_outputs = np.dot(weights, inputs) + biases
print(layer_outputs)