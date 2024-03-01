# Ate o momento temos usado um unico sample que seria uma lista de inputs, porém é mais comum usar batches
# que são lista de samples
# O motivo para usar batches é que torna mais rápido o treinamento com paralelismo e tbm ajuda na generalização
# O que é generalização?

inputs = [1, 2, 3, 2.5] #Até o momento tem sido assim
batch = [[1, 5, 6, 2],
         [3, 2, 1, 3],
         [5, 2, 1, 2],
         [6, 4, 8, 4],
         [2, 8, 5, 3],
         [1, 1, 9, 4],
         [6, 6, 0, 4],
         [8, 7, 6, 4]]

# Perceba que agora temos uma matriz de inputs e temos uma matriz de pesos. E o produto de matriz é diferente 
# do produto de vetores que estavamos fazendo anteriormente 

# Matrix Product
# Nós vamos fazer o produto das linhas da primeira matriz com as colunas da segunda matriz
# O i-th elemento da matriz resultante é o resultado da soma do produto da linha da primeira matriz com 
# a coluna da segunda matriz
# [1, 2, 3, 4] 
# [[1],
#  [2],
#  [3],
#  [4]]
# [X]
# Porem o tamanho da segunda dimensão da primeira matriz deve ser do mesmo tamanho da primeira dimensão da segunda matriz 
# Se a matriz1 é (8, 4) (8 linhas, 4 colunas) a segunda precisa ser (4, X) (4 linhas, X colunas)
# Para fazer o produto das duas matrizes a baixo precisariamos que as dimensões da segunda matriz fossem
# (4, 3)

# Shape (8, 4)
# batch = [[1, 5, 6, 2], 
#          [3, 2, 1, 3],
#          [5, 2, 1, 2],
#          [6, 4, 8, 4],
#          [2, 8, 5, 3],
#          [1, 1, 9, 4],
#          [6, 6, 0, 4],
#          [8, 7, 6, 4]]


# Na matriz a baixo as dimensões dela são (3, 4) (3 linhas, 4 colunas)
# weights = [[0.2, 0.8, -0.5, 1],
#            [0.5, -0.91, 0.26, -0.5],
#            [-0.26, -0.27, 0.17, 0.87]]
# A segunda matriz deveria ser assim (4, 3)
# weights = [[ 0.2,  0.8, -0.5],
#            [ 0.8, -0.91, 0.26],
#            [-0.5, -0.27, 0.17],
#            [ 1.0, -0.5,  0.87]]

# E o shape da matriz resultante vai ser (primeira dimensão da primeira matriz, segunda dimensão da segunda matriz)
# No exemplo acima a matriz resultante seria (8, 4)

# Para fazer o produto a quantidade de colunas da primeira matriz tem que ser a mesma quantidade de linhas da segunda matriz
