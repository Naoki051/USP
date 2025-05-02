from mlp_functions import *
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import string

def treinamento_mlp(X_treino, Y_treino, num_entrada, num_oculta, num_saida, taxa_de_aprendizado, num_iteracoes, imprimir_custo=False):
    """
    Função para treinar uma MLP simples de duas camadas.

    Argumentos:
    X_treino -- Dados de treinamento de dimensões (num_entrada, num_exemplos)
    Y_treino -- Rótulos de treinamento de dimensões (num_saida, num_exemplos)
    num_entrada -- Número de neurônios na camada de entrada
    num_oculta -- Número de neurônios na camada oculta
    num_saida -- Número de neurônios na camada de saída
    taxa_de_aprendizado -- Taxa de aprendizado para o gradiente descendente
    num_iteracoes -- Número de iterações (épocas) de treinamento
    imprimir_custo -- Booleano para indicar se o custo deve ser impresso periodicamente

    Retorna:
    parametros -- Dicionário contendo os parâmetros aprendidos (W1, b1, W2, b2)
    """

    # Inicializa os parâmetros
    parametros = inicializar_parametros(num_entrada, num_oculta, num_saida)
    custos = []

    # Loop de treinamento
    for i in range(num_iteracoes):
        # Feedforward
        A2, cache = feedforward(X_treino, parametros)

        # Calcula o custo erro quadrático médio
        custo = mean_squared_error(A2, Y_treino)
        custos.append(custo)

        # Backpropagation
        grads = backpropagation(parametros, cache, X_treino, Y_treino)

        # Atualiza os parâmetros
        parametros = atualizar_parametros(parametros, grads, taxa_de_aprendizado)

        # Imprime o custo periodicamente
        if imprimir_custo and i % 100 == 0:
            print(f"Custo após iteração {i}: {custo:.4f}")

    print("Treinamento concluído!")
    if imprimir_custo:
        plotar_custos_mlp(custos)

    return parametros

def treinamento_mlp_lento(X_treino, Y_treino, num_entrada, num_oculta, num_saida, taxa_de_aprendizado, num_iteracoes, imprimir_custo=False):
    """
    Função para treinar uma MLP simples de duas camadas processando um exemplo por vez.

    Argumentos:
    X_treino -- Dados de treinamento de dimensões (num_entrada, num_exemplos)
    Y_treino -- Rótulos de treinamento de dimensões (num_saida, num_exemplos)
    num_entrada -- Número de neurônios na camada de entrada
    num_oculta -- Número de neurônios na camada oculta
    num_saida -- Número de neurônios na camada de saída
    taxa_de_aprendizado -- Taxa de aprendizado para o gradiente descendente
    num_iteracoes -- Número de épocas de treinamento
    beta -- Hiperparâmetro do momentum (padrão: 0.9)
    imprimir_custo_a_cada -- Imprime o custo a cada este número de exemplos processados

    Retorna:
    parametros -- Dicionário contendo os parâmetros aprendidos (W1, b1, W2, b2)
    custos_por_exemplo -- Lista dos custos calculados após processar cada exemplo
    """

    parametros = inicializar_parametros(num_entrada, num_oculta, num_saida)
    custos_por_exemplo = []
    custos_medios = []
    num_exemplos = X_treino.shape[1]

    # Loop de treinamento
    for i in range(num_iteracoes):
        # Loop através de cada exemplo de entrada
        for j in range(num_exemplos):
            # Seleciona o exemplo atual
            exemplo_x = X_treino[:, j:j+1]
            exemplo_y = Y_treino[:, j:j+1]

            # Feedforward para o exemplo atual
            A2, cache = feedforward(exemplo_x, parametros)

            # Calcula o custo para este exemplo
            custo = mean_squared_error(A2, exemplo_y)
            custos_por_exemplo.append(custo)

            # Backpropagation para o exemplo atual
            grads = backpropagation(parametros, cache, exemplo_x, exemplo_y)

            # Atualiza os parâmetros
            parametros = atualizar_parametros(parametros, grads, taxa_de_aprendizado)
            custo_medio_epoca = sum(custos_por_exemplo[-num_exemplos:]) / num_exemplos
            custos_medios.append(custo_medio_epoca)
        if imprimir_custo:
            # Calcula o custo médio da época (opcional, para acompanhamento)
            
            print(f"Custo médio após {i} iteracoes: {custo_medio_epoca:.6f}")
    
    print("Treinamento concluído (um exemplo por vez)!")

    if imprimir_custo:
        plotar_custos_mlp(custos_medios)
    return parametros

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_derivada(s):
    return s * (1 - s)

def relu(z):
    return np.maximum(0, z)

def relu_derivada(a):
    return np.int64(a > 0)

def inicializar_parametros(num_entrada, num_oculta, num_saida):
    parametros = {
        "W1": np.random.randn(num_oculta, num_entrada) * 0.01,
        "b1": np.zeros((num_oculta, 1)),
        "W2": np.random.randn(num_saida, num_oculta) * 0.01,
        "b2": np.zeros((num_saida, 1))
    }
    return parametros

def feedforward(X, parametros):
    W1 = parametros["W1"]
    b1 = parametros["b1"]
    W2 = parametros["W2"]
    b2 = parametros["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}
    return A2, cache

def backpropagation(parametros, cache, X, Y):
    m = X.shape[1]

    W1 = parametros["W1"]
    W2 = parametros["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    Z1 = cache["Z1"]

    dZ2 = A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * relu_derivada(Z1)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}
    return grads

def atualizar_parametros(parametros, grads, taxa_de_aprendizado):
    parametros["W1"] -= taxa_de_aprendizado * grads["dW1"]
    parametros["b1"] -= taxa_de_aprendizado * grads["db1"]
    parametros["W2"] -= taxa_de_aprendizado * grads["dW2"]
    parametros["b2"] -= taxa_de_aprendizado * grads["db2"]
    return parametros


def prever(X, parametros):
    A2, cache = feedforward(X, parametros)
    return A2

def plotar_custos_mlp(custos_historico, label='Custo'):
    plt.plot(range(len(custos_historico)), custos_historico, label=label)
    plt.xlabel('Número de Iterações/Épocas')
    plt.ylabel('Custo')
    plt.title('Histórico do Custo durante o Treinamento')
    plt.grid(True)
    plt.legend()
    plt.show()

def treinamento_mlp_com_validacao(X_treino, Y_treino, X_val, Y_val, num_entrada, num_oculta, num_saida, taxa_de_aprendizado, num_iteracoes, imprimir_custo=False):
    """
    Função para treinar uma MLP simples de duas camadas processando um exemplo por vez,
    com avaliação no conjunto de validação a cada época.

    Argumentos:
    X_treino -- Dados de treinamento de dimensões (num_entrada, num_exemplos_treino)
    Y_treino -- Rótulos de treinamento de dimensões (num_saida, num_exemplos_treino)
    X_val -- Dados de validação de dimensões (num_entrada, num_exemplos_val)
    Y_val -- Rótulos de validação de dimensões (num_saida, num_exemplos_val)
    num_entrada -- Número de neurônios na camada de entrada
    num_oculta -- Número de neurônios na camada oculta
    num_saida -- Número de neurônios na camada de saída
    taxa_de_aprendizado -- Taxa de aprendizado para o gradiente descendente
    num_iteracoes -- Número de épocas de treinamento
    imprimir_custo -- Booleano para indicar se o custo deve ser impresso periodicamente

    Retorna:
    parametros -- Dicionário contendo os parâmetros aprendidos (W1, b1, W2, b2)
    custos_treino_medio -- Lista dos custos médios de treinamento por época
    custos_val -- Lista dos custos de validação por época
    """

    parametros = inicializar_parametros(num_entrada, num_oculta, num_saida)
    custos_treino_por_exemplo = []
    custos_treino_medio = []
    custos_val = []
    num_exemplos_treino = X_treino.shape[1]
    num_exemplos_val = X_val.shape[1]

    # Loop de treinamento (épocas)
    for i in range(num_iteracoes):
        # Permutar os dados de treino para evitar overfitting
        permutacao = np.random.permutation(num_exemplos_treino)
        X_treino = X_treino[:, permutacao]
        Y_treino = Y_treino[:, permutacao]

        for j in range(num_exemplos_treino):
            # Seleciona o exemplo atual
            exemplo_x_treino = X_treino[:, j:j+1]
            exemplo_y_treino = Y_treino[:, j:j+1]

            # Feedforward para o exemplo atual
            A2_treino, cache_treino = feedforward(exemplo_x_treino, parametros)

            # Calcula o custo para este exemplo de treinamento
            custo_treino = mean_squared_error(A2_treino, exemplo_y_treino)
            custos_treino_por_exemplo.append(custo_treino)

            # Backpropagation para o exemplo atual
            grads_treino = backpropagation(parametros, cache_treino, exemplo_x_treino, exemplo_y_treino)

            # Atualiza os parâmetros
            parametros = atualizar_parametros(parametros, grads_treino, taxa_de_aprendizado)

        # Calcula o custo médio de treinamento para a época
        custo_medio_epoca = sum(custos_treino_por_exemplo[-num_exemplos_treino:]) / num_exemplos_treino
        custos_treino_medio.append(custo_medio_epoca)

        # Avaliação no conjunto de validação ao final de cada época
        A2_val, _ = feedforward(X_val, parametros)
        custo_valid = mean_squared_error(A2_val, Y_val)
        custos_val.append(custo_valid)

        if imprimir_custo:
            print(f"Época {i}, Custo Treino Médio: {custo_medio_epoca:.6f}, Custo Validação: {custo_valid:.6f}")

    print("Treinamento concluído (um exemplo por vez com validação)!")

    if imprimir_custo:
        plotar_custos_com_validacao(custos_treino_medio,custos_val)

    return parametros, custos_treino_medio, custos_val

def treinamento_mlp_com_momentum(X_treino, Y_treino, num_entrada, num_oculta, num_saida, taxa_de_aprendizado, num_iteracoes, beta=0.9, imprimir_custo=False):
    """
    Função para treinar uma MLP simples de duas camadas com momentum.

    Argumentos:
    X_treino -- Dados de treinamento de dimensões (num_entrada, num_exemplos)
    Y_treino -- Rótulos de treinamento de dimensões (num_saida, num_exemplos)
    num_entrada -- Número de neurônios na camada de entrada
    num_oculta -- Número de neurônios na camada oculta
    num_saida -- Número de neurônios na camada de saída
    taxa_de_aprendizado -- Taxa de aprendizado para o gradiente descendente
    num_iteracoes -- Número de iterações (épocas) de treinamento
    beta -- Hiperparâmetro do momentum (padrão: 0.9)
    imprimir_custo -- Booleano para indicar se o custo deve ser impresso periodicamente

    Retorna:
    parametros -- Dicionário contendo os parâmetros aprendidos (W1, b1, W2, b2)
    custos -- Lista dos custos calculados em cada iteração
    """

    parametros, velocidades = inicializar_parametros_e_velocidades(num_entrada, num_oculta, num_saida)
    custos = []

    # Loop de treinamento
    for i in range(num_iteracoes):
        # Feedforward
        A2, cache = feedforward(X_treino, parametros)

        # Calcula o custo
        custo = mean_squared_error(A2, Y_treino)
        custos.append(custo)

        # Backpropagation
        grads = backpropagation(parametros, cache, X_treino, Y_treino)

        # Atualiza os parâmetros com momentum
        parametros, velocidades = atualizar_parametros_com_momentum(parametros, grads, velocidades, taxa_de_aprendizado, beta)

        # Imprime o custo periodicamente
        if imprimir_custo and i % 1000 == 0:
            print(f"Custo após iteração {i}: {custo:.4f}")

    print("Treinamento concluído com momentum!")
    return parametros, custos

def treinamento_mlp_com_momentum_lento(X_treino, Y_treino, num_entrada, num_oculta, num_saida, taxa_de_aprendizado, num_iteracoes, beta=0.9, imprimir_custo=False):
    """
    Função para treinar uma MLP simples de duas camadas com momentum.

    Argumentos:
    X_treino -- Dados de treinamento de dimensões (num_entrada, num_exemplos)
    Y_treino -- Rótulos de treinamento de dimensões (num_saida, num_exemplos)
    num_entrada -- Número de neurônios na camada de entrada
    num_oculta -- Número de neurônios na camada oculta
    num_saida -- Número de neurônios na camada de saída
    taxa_de_aprendizado -- Taxa de aprendizado para o gradiente descendente
    num_iteracoes -- Número de iterações (épocas) de treinamento
    beta -- Hiperparâmetro do momentum (padrão: 0.9)
    imprimir_custo -- Booleano para indicar se o custo deve ser impresso periodicamente

    Retorna:
    parametros -- Dicionário contendo os parâmetros aprendidos (W1, b1, W2, b2)
    custos -- Lista dos custos calculados em cada iteração
    """

    parametros, velocidades = inicializar_parametros_e_velocidades(num_entrada, num_oculta, num_saida)
    num_exemplos = X_treino.shape[1]
    custos_por_exemplo = []
    custos_medios =[]
    # Loop de treinamento
    for i in range(num_iteracoes):
       # Loop através de cada exemplo de entrada
        for j in range(num_exemplos):
            # Seleciona o exemplo atual
            exemplo_x = X_treino[:, j:j+1]
            exemplo_y = Y_treino[:, j:j+1]

            # Feedforward para o exemplo atual
            A2, cache = feedforward(exemplo_x, parametros)

            # Calcula o custo para este exemplo
            custo = mean_squared_error(A2, exemplo_y)
            custos_por_exemplo.append(custo)

            # Backpropagation para o exemplo atual
            grads = backpropagation(parametros, cache, exemplo_x, exemplo_y)

            # Atualiza os parâmetros
            parametros, velocidades = atualizar_parametros_com_momentum(parametros, grads, velocidades, taxa_de_aprendizado, beta)
            custo_medio_epoca = sum(custos_por_exemplo[-num_exemplos:]) / num_exemplos
            custos_medios.append(custo_medio_epoca)
        if imprimir_custo and i % 10 == 0:
            # Calcula o custo médio da época (opcional, para acompanhamento)
            
            print(f"Custo médio após {i} iteracoes: {custo_medio_epoca:.6f}")
    if imprimir_custo:
        plotar_custos_mlp(custos_medios)

    print("Treinamento concluído com momentum!")
    return parametros, custos_medios

def prever(X, parametros):
    A2, cache = feedforward(X, parametros)
    return A2

def plotar_custos_mlp(custos_historico):
    plt.plot(range(len(custos_historico)), custos_historico)
    plt.xlabel('Número de Iterações')
    plt.ylabel('Custo')
    plt.title('Histórico do Custo durante o Treinamento')   
    plt.grid(True)
    plt.show()

def plotar_custos_com_validacao(custos_treino, custos_val):
    """Plota os custos de treinamento e validação ao longo das épocas."""
    plt.plot(range(len(custos_treino)), custos_treino, label='Custo Treino Médio')
    plt.plot(range(len(custos_val)), custos_val, label='Custo Validação')
    plt.xlabel('Número de Épocas')
    plt.ylabel('Custo')
    plt.title('Histórico do Custo durante o Treinamento com Validação')
    plt.grid(True)
    plt.legend()
    plt.show()

def plotar_matriz_confusao(letras_esperadas, letras_previstas):
    """Plota a matriz de confusão."""
    alfabeto = list(string.ascii_uppercase)  # Converte a string para uma lista de caracteres
    cm = confusion_matrix(letras_esperadas, letras_previstas, labels=alfabeto)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=alfabeto, yticklabels=alfabeto)
    plt.xlabel('Letras Previstas')
    plt.ylabel('Letras Esperadas')
    plt.title('Matriz de Confusão')
    plt.show()