# Página 20

# O que são os inputs?
# Inputs são valores que irão passar pela nossa rede neural e irão resultar em um output, sendo esse output um valor
# desejado ou não. Imagine uma foto de 28x28 pixels e que nela esta um digito escrito a mão, os nossos inputs serão
# os valores numa escala de 0 à 1, sendo 1 o mais branco possível e 0 o mais preto possível. Esses valores vão passar 
# na nossa rede neural e nos dará uma saída, dizendo qual número está escrito na nossa imgs (de 0 até 9). 
# Esse é um exemplo do que nossa rede neural pode fazer, ela pode identificar coisas numa img... 
# O que vai acontecer com esses inputs?
# Eles serão multiplicados pelos pesos que os ligam à um neuronio da rede neural, que resultara numa saida e essa saida
# sera o input de uma proxima camada da nossa rede neural.
inputs = [1, 2, 3]
# O que são os pesos?
# Imagine que nossos neuronios terão ligação com cada um de nossos inputs, os pesos são os valores dessas ligações do 
# neuronio com o input. O nosso array abaixo é um neuronio que tem ligação com cada um dos inputs do nosso array acimam.
weights = [0.2, 0.8, -0.5]
# O que é um bias?
# Pq tem apenas um bias?
# Existe apenas um bias, pq o bias é um valor que é somado com o produto do nosso peso com o input e ele está ligado 
# ao nosso neuronio. Como temos um neuronio, teremos apenas um bias
bias = 2
# E se eu colocar mais um input no meu array de inputs?
# Se colocarmos mais um input no nosso array de inputs, teremos que colocar mais um peso em nosso array de pesos, para
# que assim o nosso input tenha uma ligação ao nosso neuronio.
inputs.append(2.5)
weights.append(1.0)
# Qual o calculo feito agora?
# Nos fazemos o seguinte calculo inputs[n] * weights[n] + bias

# OBS: Fiz dessa forma mais "descritiva" apenas para ficar bem claro como ocorrem as coisas 
# o código não vai ser escrito assim na mão
output = inputs[0] * weights[0] + inputs[1] * weights[1] + inputs[2] * weights[2] + inputs[3] * weights[3] + bias
# Esse nosso calculo vai resultar no na saida 4.8 e pode ser a entrada de outro neuronio ou a saida da nossa
# camada de saida da nossa rede neural e representar o valor que queriamos ou não
output2 = []
value = 0
for i in range(len(inputs)):
    value+=(inputs[i] * weights[i])
output2.append(value + bias)

print("First Neuron output:",output)
print("First Neuron output:",output2)