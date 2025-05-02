import string
import numpy as np
from training import treinamento_mlp_com_validacao, prever, mean_squared_error, plotar_matriz_confusao

def probabilidades_para_one_hot(probabilidades):
    indice_max = np.argmax(probabilidades, axis=0)
    one_hot = np.zeros_like(probabilidades)
    one_hot[indice_max, np.arange(probabilidades.shape[1])] = 1
    return one_hot

def lista_de_letras_para_one_hot(letras):
    alfabeto = string.ascii_uppercase
    tamanho_alfabeto = len(alfabeto)
    mapa_letras = {letra: idx for idx, letra in enumerate(alfabeto)}

    indices = [mapa_letras[letra] for letra in letras if letra in mapa_letras]
    matriz = np.zeros((len(indices), tamanho_alfabeto), dtype=int)
    matriz[np.arange(len(indices)), indices] = 1
    return matriz

def one_hot_matriz_para_letras(matriz_one_hot):
    alfabeto = string.ascii_uppercase
    if matriz_one_hot.ndim != 2 or matriz_one_hot.shape[1] != len(alfabeto):
        print("Erro: matriz inválida para conversão.")
        return None
    indices = np.argmax(matriz_one_hot, axis=1)
    somas = np.sum(matriz_one_hot, axis=1)
    letras = [alfabeto[i] if soma == 1 else None for i, soma in zip(indices, somas)]
    return [letra for letra in letras if letra is not None]

def carregar_entradas(caminho):
    # Carregar os dados, ignorando entradas vazias
    entradas = np.genfromtxt(caminho, delimiter=',', dtype=int, filling_values=0)
    
    # Remover colunas e/ou linhas com valores nulos (0) ou vazios
    entradas = entradas[:, ~np.all(entradas == 0, axis=0)]  # Remove colunas que são inteiramente 0
    entradas = entradas[~np.all(entradas == 0, axis=1)]  # Remove linhas que são inteiramente 0

    return entradas.T


def carregar_saidas(caminho):
    with open(caminho, "r") as arq:
        return [linha.strip() for linha in arq if linha.strip()]

# Carregamento e preparação dos dados
X = carregar_entradas("data/X.txt")
Y_letras = carregar_saidas("data/Y_letra.txt")
Y = lista_de_letras_para_one_hot(Y_letras).T

# Divisão dos dados
X_train, X_val, X_test = X[:, :-260], X[:, -260:-130], X[:, -130:]
Y_train, Y_val, Y_test = Y[:, :-260], Y[:, -260:-130], Y[:, -130:]

# Hiperparâmetros
taxa_de_aprendizado = 0.01
num_iteracoes = 100
imprimir_custo = True

# Treinamento
parametros_aprendidos, custos_treino, custos_val = treinamento_mlp_com_validacao(
    X_train, Y_train, X_val, Y_val,
    num_entrada=120, num_oculta=64, num_saida=26,
    taxa_de_aprendizado=taxa_de_aprendizado,
    num_iteracoes=num_iteracoes,
    imprimir_custo=imprimir_custo
)

# Avaliação
previsoes = prever(X_test, parametros_aprendidos)
previsoes_one_hot = probabilidades_para_one_hot(previsoes)

acertos = np.sum(np.argmax(previsoes_one_hot, axis=0) == np.argmax(Y_test, axis=0))
total = Y_test.shape[1]
print(f"Acurácia no teste: {acertos / total * 100:.2f}%")

# Conversão para letras
letras_previstas = one_hot_matriz_para_letras(previsoes_one_hot.T)
letras_esperadas = one_hot_matriz_para_letras(Y_test.T)

if letras_previstas and letras_esperadas:
    plotar_matriz_confusao(letras_esperadas, letras_previstas)
else:
    print("Erro na conversão de matrizes one-hot para letras.")
