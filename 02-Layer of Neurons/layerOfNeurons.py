# Nossa rede neural não terá apenas um neuronio, mas sim vários, milhares ou milhoes de neuronios por camadas. 
# E ela pode ter varias camadas, a camada que recebe o valor dos inputs, as camadas mais internas que recebem
# os outputs de camadas anteriores e a camada de saida que resultará em um saída, sendo essa saída que queremos ou não.
# Vamos pegar a camada que recebe o valor dos inputs, cada neuronio dessa camada ter um peso associado a cada um dos 
# nosso inputs e cada neuronio tera o seu próprio bias.
# No exemplo a baixo temos uma camada com três neuronios e cada um dos neuronios possui um peso ligado a um dos inputs. 
# Isso é chamado de fully connected neuron networks. Você pode criar redes neurais que não são totalmente conectadas
inputs = [1, 2, 3, 2.5]
weights1 = [0.2, 0.8, -0.5, 1] #Neuronio 1 e os pesos que ligam a cada um dos inputs
weights2 = [0.5, -0.91, 0.26, -0.5] #Neuronio 2 e os pesos que ligam a cada um dos inputs
weights3 = [-0.26, -0.27, 0.17, 0.87] #Neuronio 3 e os pesos que ligam a cada um dos inputs
biases = [2, 3, 0.5]
weights = [[0.2, 0.8, -0.5, 1],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]
# OBS: Fiz dessa forma mais "descritiva" apenas para ficar bem claro como ocorrem as coisas 
# o código não vai ser escrito assim na mão
outputs = [
    inputs[0] * weights1[0] + inputs[1] * weights1[1] + inputs[2] * weights1[2] + inputs[3] * weights1[3] + biases[0],
    inputs[0] * weights2[0] + inputs[1] * weights2[1] + inputs[2] * weights2[2] + inputs[3] * weights2[3] + biases[1],
    inputs[0] * weights3[0] + inputs[1] * weights3[1] + inputs[2] * weights3[2] + inputs[3] * weights3[3] + biases[2]
]
# Esse nosso calculo vai resultar na saida [4.8, 1.21, 2.385] e pode ser a entrada para outra camada de neuronios ou a saida da nossa camada de saida da nossa rede neural e representar o valor que queriamos ou não

# Minha forma
outputs2 = []
for i in range(len(weights)):
    value = 0
    for j in range(len(weights[i])):
        value+=(inputs[j] * weights[i][j])
    outputs2.append(value + biases[i])

# Forma do livro (Assim aprendo mais sobre a linguagem python)
layer_outputs = []
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight
    layer_outputs.append(neuron_output + neuron_bias)

print("Layer of neurons outputs: ", outputs)
print("Layer of neurons outputs2: ", outputs2)
print("Layer of neurons layer_outputs: ", layer_outputs)