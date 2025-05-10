import numpy as np

# Funções de ativação e derivadas
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivada(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivada(x):
    return (x > 0).astype(float)

def tanh(x):
    return np.tanh(x)

def tanh_derivada(x):
    return 1 - np.tanh(x) ** 2

def novo_modelo(entrada_dim, ocultas_dim, saida_dim, ativacao_oculta=tanh, ativacao_saida=sigmoid):
    modelo = [
        {
            'pesos': np.random.randn(entrada_dim, ocultas_dim),
            'bias': np.zeros((1, ocultas_dim)),
            'funcao_ativacao': ativacao_oculta
        },
        {
            'pesos': np.random.randn(ocultas_dim, saida_dim),
            'bias': np.zeros((1, saida_dim)),
            'funcao_ativacao': ativacao_saida
        }
    ]
    return modelo


def feedforward(camada, entrada):
    funcao_camada = camada['funcao_ativacao']
    somatorio_entrada = np.dot(entrada, camada['pesos']) + camada['bias']
    camada['somatorio_entrada'] = somatorio_entrada
    if funcao_camada == sigmoid:
        return sigmoid(somatorio_entrada)
    elif funcao_camada == relu:
        return relu(somatorio_entrada)
    elif funcao_camada == tanh:
        return tanh(somatorio_entrada)
    else:
        raise ValueError(f"Função de ativação não reconhecida: {funcao_camada}")

def backpropagation(camada, grad_recebido):
    funcao_camada = camada['funcao_ativacao']
    somatorio_entrada = camada['somatorio_entrada']

    if funcao_camada == sigmoid:
        derivada = sigmoid_derivada(somatorio_entrada)
    elif funcao_camada == relu:
        derivada = relu_derivada(somatorio_entrada)
    elif funcao_camada == tanh:
        derivada = tanh_derivada(somatorio_entrada)
    else:
        raise ValueError(f"Função de ativação não reconhecida: {funcao_camada}")

    grad_propag = grad_recebido * derivada
    return grad_propag

def atualizacao_camadas(camada, taxa_aprendizado):
    camada['pesos'] -= taxa_aprendizado * camada['grad_pesos']
    camada['bias']  -= taxa_aprendizado * camada['grad_bias']

# Etapa de treino
def train_step(entradas, saidas_esperadas, modelo, taxa_aprendizado):
    total_loss = 0
    for i in range(len(entradas)):
        entrada = entradas[i].reshape(1, -1)
        saida_esperada = saidas_esperadas[i].reshape(1, -1)
        # Feedforward
        for camada in modelo:
            camada['entrada'] = entrada
            saida = feedforward(camada, entrada)
            camada['saida'] = saida
            entrada = saida
        total_loss += np.mean((saida - saida_esperada) ** 2)
        # Backpropagation
        grad_propag = 2*(saida - saida_esperada)
        for camada in reversed(modelo):
            delta = backpropagation(camada, grad_propag)
            entrada_anterior = camada['entrada']
            camada['grad_pesos'] = np.dot(entrada_anterior.T, delta)
            camada['grad_bias'] = delta
            grad_propag = np.dot(delta, camada['pesos'].T)
        # Atualização dos pesos
        for camada in modelo:
            atualizacao_camadas(camada, taxa_aprendizado)
    return total_loss / len(entradas)
