import numpy as np
# Transposição de matriz é uma operação que transforma as linhas de uma matriz em colunas 
# e suas colunas em linhas 
# matriz_não_transposta = [[1, 2, 3],
#                          [4, 5, 6],
#                          [7, 8, 9]]
# # Transposição da matriz acima
# matriz_transposta = [[1, 4, 7],
#                      [2, 5, 8],
#                      [3, 6, 9]]


a = [1, 2, 3]
b = [2, 3, 4]

a = np.array([a]) #Matriz (1, n)
b = np.array([b]).T #Matriz (n, 1)

# E para fazer o produto de matrizes precisamos que a segunda dimensão da primeira matriz seja igual a 
# ao tamanho da primeira dimensão da segunda matriz 
# E a matriz resultante tera (x1, x2) sendo x1 a primeira dimensão da primeira matriz e x2 a segunda dimensão 
# da segunda matriz
print(np.dot(a, b))

# Fazendo o produto de matrizes em uma camada de neuronios e um lote de dados (batch de inputs)
inputs = [[1.0,  2.0,  3.0, 2.5],
          [2.0,  5.0, -1.0, 2.0],
          [-1.5, 2.7,  3.3, -0.8]]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]
inputs = np.array(inputs)
weights = np.array(weights)



# Se eu fizer o produto das matrizes acima vai dar erro de shape, por causa da regra para 
# fazer tal operação. Para isso nós devemos fazer a transposição da segunda matriz weights

print(np.dot(inputs, weights.T) + biases)

# Resultado será, onde cada coluna será correspondente a um neuronio
# [[ 4.8    1.21   2.385]
#  [ 8.9   -1.81   0.2  ]
#  [ 1.41   1.051  0.026]]
