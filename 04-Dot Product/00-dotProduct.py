# Dot product é a multiplicação de dois vetores (produto escalar)
# multiplicamos o valor de um elemento de um vetor pelo elemento do outro vetor 
# de mesmo indice e depois adicionamos o resultado com a multiplicação dos elementos de 
# outro indice 

a = [1, 2, 3]
b = [2, 3, 4]
result = a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
print("Result sem numpy: ", result)

# Podemos fazer essa operação usando um laço for ou usando a lib Numpy
import numpy as np

# Agora ao inves de chamar de a e b, vamos chamar a de inputs e b de weights
# se parece muito com as operações que estavamos fazendo no primeiro neuronio e 
# e camadas de neuronios (falta apenas adicionar o bias)

inputs = [1, 2, 3]
weights = [2, 3, 4]
output = np.dot(inputs, weights)

print("With .dot() do numpy: ", output)