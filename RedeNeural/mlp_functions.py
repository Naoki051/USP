import numpy as np

def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

def sigmoid_derivada(s):
    ds = s * (1 - s)
    return ds

def tanh(z):
    t = np.tanh(z)
    return t

def tanh_derivada(t):
    dt = 1 - np.power(t, 2)
    return dt

def relu(z):
    a = np.maximum(0, z)
    return a

def relu_derivada(z):
    dz = np.int64(z > 0)
    return dz

def inicializar_parametros(num_entrada, num_oculta, num_saida):
    """
    Inicializa os pesos e biases para uma rede neural de duas camadas.

    Argumentos:
    num_entrada -- tamanho da camada de entrada
    num_oculta -- tamanho da camada oculta
    num_saida -- tamanho da camada de saída

    Retorna:
    parametros -- dicionário Python contendo os parâmetros:
                        W1 -- matriz de pesos da camada 1, de dimensões (num_oculta, num_entrada)
                        b1 -- vetor de bias da camada 1, de dimensões (num_oculta, 1)
                        W2 -- matriz de pesos da camada 2, de dimensões (num_saida, num_oculta)
                        b2 -- vetor de bias da camada 2, de dimensões (num_saida, 1)
    """

    # Inicializa os pesos aleatoriamente com valores pequenos
    W1 = np.random.randn(num_oculta, num_entrada) * 0.01
    W2 = np.random.randn(num_saida, num_oculta) * 0.01

    # Inicializa os biases com zeros
    b1 = np.zeros((num_oculta, 1))
    b2 = np.zeros((num_saida, 1))

    parametros = {
        "W1": W1,
        "b1": b1,
        "W2": W2,
        "b2": b2
    }

    return parametros

def inicializar_parametros_e_velocidades(num_entrada, num_oculta, num_saida):
    """
    Inicializa os parâmetros (pesos e biases) e as velocidades com zeros.

    Argumentos:
    num_entrada -- Número de neurônios na camada de entrada
    num_oculta -- Número de neurônios na camada oculta
    num_saida -- Número de neurônios na camada de saída

    Retorna:
    parametros -- Dicionário contendo os parâmetros da rede (W1, b1, W2, b2)
    velocidades -- Dicionário contendo as velocidades iniciais (dW1, db1, dW2, db2)
    """
    parametros = inicializar_parametros(num_entrada, num_oculta, num_saida)
    velocidades = {}
    for chave in parametros:
        velocidades["d" + chave] = np.zeros_like(parametros[chave])

    return parametros, velocidades

def print_parametros(parametros):
    """
    Imprime o formato (shape) das matrizes de pesos e vetores de bias
    contidos no dicionário de parâmetros.

    Argumentos:
    parametros -- dicionário Python contendo os parâmetros da rede neural
                  (W1, b1, W2, b2).
    """
    print("Formato de W1:", parametros["W1"].shape)
    print("Formato de b1:", parametros["b1"].shape)
    print("Formato de W2:", parametros["W2"].shape)
    print("Formato de b2:", parametros["b2"].shape)

def feedforward(X, parametros):
    """
    Implementa a propagação para frente para uma rede neural de duas camadas.

    Argumentos:
    X -- dados de entrada de dimensões (num_entrada, num_exemplos)
    parametros -- dicionário Python contendo os parâmetros:
                  "W1", "b1", "W2", "b2"

    Retorna:
    A2 -- A saída da camada de saída (ativações)
    cache -- um dicionário contendo "Z1", "A1", "Z2", "A2";
             armazenados para serem usados na propagação para trás
    """
    # Recupera os parâmetros do dicionário "parametros"
    W1 = parametros["W1"]
    b1 = parametros["b1"]
    W2 = parametros["W2"]
    b2 = parametros["b2"]

    # Camada oculta
    Z1 = np.dot(W1, X) + b1
    A1 = tanh(Z1)

    # Camada saída
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    return A2, cache

def print_cache(cache):
    """
    Imprime o formato (shape) e os valores dos elementos armazenados
    no dicionário 'cache' para a etapa de backpropagation.
    """
    print("\nCache:")
    for key, value in cache.items():
        print(f"Formato de {key}:", value.shape)
        print(f"{key}:\n", value)

def backpropagation(parametros, cache, X, Y):
    """
    Implementa a propagação para trás para uma rede neural de duas camadas.

    Argumentos:
    parametros -- dicionário contendo os parâmetros da rede (W1, b1, W2, b2)
    cache -- dicionário contendo as saídas intermediárias (Z1, A1, Z2, A2)
    X -- dados de entrada de dimensões (num_entrada, num_exemplos)
    Y -- rótulos "verdadeiros" de dimensões (num_saida, num_exemplos)

    Retorna:
    grads -- dicionário contendo os gradientes em relação aos diferentes parâmetros
             (dW1, db1, dW2, db2)
    """
    
    # Recupera W1, W2 do dicionário de parâmetros
    W2 = parametros["W2"]

    # Recupera A1, A2 do dicionário de cache
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    # Cálculo do gradiente da camada de saída 
    dZ2 = (A2 - Y) * sigmoid_derivada(A2)
    dW2 = np.dot(dZ2, A1.T)
    db2 = np.sum(dZ2, axis=1, keepdims=True) # Soma ao longo dos exemplos

    # Cálculo do gradiente da camada oculta
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * tanh_derivada(A1)
    dW1 = np.dot(dZ1, X.T)  
    db1 = np.sum(dZ1, axis=1, keepdims=True) # Soma ao longo dos exemplos

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

    return grads

def mean_squared_error(predictions, labels):
    """Calcula o erro quadrático médio."""

    loss = np.mean((labels - predictions)**2)
    return loss

def mean_absolute_error(predictions, labels):
    """Calcula o erro absoluto médio."""

    loss = np.mean(np.abs(labels - predictions))
    return loss

def atualizar_parametros(parametros, grads, taxa_de_aprendizado):
    """
    Atualiza os parâmetros da rede neural usando a regra do gradiente descendente.

    Argumentos:
    parametros -- dicionário contendo os parâmetros da rede (W1, b1, W2, b2)
    grads -- dicionário contendo os gradientes (dW1, db1, dW2, db2)
    taxa_de_aprendizado -- a taxa de aprendizado para a atualização

    Retorna:
    parametros -- dicionário contendo os parâmetros atualizados
    """

    W1 = parametros["W1"]
    b1 = parametros["b1"]
    W2 = parametros["W2"]
    b2 = parametros["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Atualiza cada parâmetro usando a regra do gradiente descendente
    W1 = W1 - taxa_de_aprendizado * dW1
    b1 = b1 - taxa_de_aprendizado * db1
    W2 = W2 - taxa_de_aprendizado * dW2
    b2 = b2 - taxa_de_aprendizado * db2

    parametros_atualizados = {"W1": W1,
                              "b1": b1,
                              "W2": W2,
                              "b2": b2}

    return parametros_atualizados

def atualizar_parametros_com_momentum(parametros, grads, v, taxa_de_aprendizado, beta):
    """
    Atualiza os parâmetros da rede neural usando o gradiente descendente com momentum.

    Argumentos:
    parametros -- dicionário contendo os parâmetros da rede (W1, b1, W2, b2)
    grads -- dicionário contendo os gradientes (dW1, db1, dW2, db2)
    v -- dicionário contendo as velocidades anteriores (dW1, db1, dW2, db2)
    taxa_de_aprendizado -- a taxa de aprendizado para a atualização
    beta -- o hiperparâmetro do momentum

    Retorna:
    parametros -- dicionário contendo os parâmetros atualizados
    v -- dicionário contendo as velocidades atualizadas
    """

    parametros_atualizados = {}
    v_atualizado = {}

    for chave in parametros:
        dW = grads["d" + chave] 
        
        # Atualizar as velocidades: vdW[l] ​= β * vdW[l] + (1-β) * dW[l]
        v["d" + chave] = beta * v["d" + chave] + (1 - beta) * dW

        # Atualizar os parâmetros: W[l] = W[l] − α * vdW[l] ​
        parametros_atualizados[chave] = parametros[chave] - taxa_de_aprendizado * v["d" + chave]

        # Para clareza do código
        v_atualizado["d" + chave] = v["d" + chave]

    return parametros_atualizados, v_atualizado