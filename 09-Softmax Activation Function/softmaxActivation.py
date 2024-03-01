# Função de ativação softmax 
# Podemos escolher diferentes funções de ativação para cada tipo de tarefa que queremos que nossa 
# NA exerça. No nosso caso queremos que ela seja uma classificadora e uma das funções comumente usadas 
# para a camada de saida é a softmax. 
# Não será usada a função ReLU para a nossa camada de saída pq com ela os valores não são normalizados
# ou seja, eles podem ser qualquer valor. Ex.: [12, 99, 82]
# e cada um dos outputs não tem ligação entre eles

# Em classificação queremos que nosso modelo nos retorne uma predição de qual classe o nosso input representa 
# E essa predição é dada em forma de um vetor, onde o seu tamanho é igual a quantidade de classes que queremos 
# classificar, e os seus valores são uma distribuição probabilistica (confidence score) de até 1 
# A nossa predição será a classe com maior pontuação
# Ex.:
# [0.45, 0.55]
# (Indice 0 igual a primeira classe e indice 1 a segunda classe)
# A predição do nosso modelo indica que o input representa a segunda classe, pq é o valor com maior pontuação 
# porem, note que a confiança do nosso modelo é baixa, pois a primeira classe ainda está com um valor alto
# Essa distribuição retornada pela função softmax é chamada de pontuação de confiança (confidence score)

# Formula da função softmax 
# e**z,i,j / sum(e**z,i,j)

# No numerador
# e = numero de euler (mais ou menos 2.71828182846) 
# z,i,j = z é um unico valor que se encontra na linha (i) e na coluna (j) daquela amostra 
# No denominador 
# Fazemos a soma das exponenciações dos valores daquela amostra

# Ex.:
import numpy as np
layer_output = [4.8, 1.21, 2.385]

E = 2.71828182846

# Exponenciando os valores
# Sem usar numpy
exp_values = []
for output in layer_output:
    exp_values.append(E**output)
print(exp_values)

# Normalizando os valores
norm_base = sum(exp_values)
normalized_values = []
for value in exp_values:
    normalized_values.append(value / norm_base)
print(normalized_values)

# Usando numpy
exp_values = np.exp(layer_output - np.max(layer_output))
print('Numero de euler exponenciado por cada um dos valores da amostra: ')
print(exp_values)
print()
norm_values = exp_values / np.sum(exp_values)
print('Valores normalizados')
print('Onde foi aplicada a formula da softmax dita acima')
print("Norma values: ", norm_values)
# Note que os valores normalizados é uma distribuição entre as classes de ate 1, se eu soma-los eles vão dar 1
# E a que tiver a maior pontuação é a nossa predição
# A forma como fizemos ate agora lida com apenas um input, porem trabalharemos com batches


# Usando batches
# layer_outputs = np.array([[4.8, 1.21, 2.385],
#                           [8.9, -1.81, 0.2],
#                           [1.41, 1.051, 0.026]])
# print(np.sum(layer_outputs, axis=None))

# print(np.sum(layer_outputs, axis=1, keepdims=True))