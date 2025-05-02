import string
import numpy as np
from training import treinamento_mlp_lento,prever

def lista_de_letras_para_one_hot(lista_de_letras):
    """
    Converte uma lista de letras maiúsculas do alfabeto em uma matriz one-hot.

    Args:
        lista_de_letras (list): Uma lista de strings, onde cada string é uma
                                 letra maiúscula do alfabeto.

    Returns:
        np.array: Uma matriz 2D NumPy onde cada linha é a codificação one-hot
                  da letra correspondente na lista de entrada. Letras inválidas
                  são ignoradas.
    """
    alfabeto = string.ascii_uppercase
    tamanho_alfabeto = len(alfabeto)
    matriz_one_hot = []
    for letra in lista_de_letras:
        if len(letra) == 1 and letra in alfabeto:
            indice = alfabeto.index(letra)
            encoding = np.zeros(tamanho_alfabeto, dtype=int)
            encoding[indice] = 1
            matriz_one_hot.append(encoding)
        else:
            print(f"Aviso: '{letra}' não é uma letra maiúscula válida e foi ignorada.")
    return np.array(matriz_one_hot)

def one_hot_matriz_para_letras(matriz_one_hot):
    """
    Converte uma matriz one-hot de codificações do alfabeto de volta para uma lista de letras.

    Args:
        matriz_one_hot (np.array): Uma matriz 2D NumPy onde cada linha é a
                                    codificação one-hot de uma letra maiúscula.

    Returns:
        list: Uma lista de letras maiúsculas correspondentes às codificações
              one-hot na matriz de entrada. Retorna None se a matriz tiver
              dimensão incorreta ou contiver codificações inválidas.
    """
    alfabeto = string.ascii_uppercase
    tamanho_alfabeto = len(alfabeto)
    lista_de_letras = []

    if matriz_one_hot.ndim != 2 or matriz_one_hot.shape[1] != tamanho_alfabeto:
        print(f"Erro: A matriz one-hot deve ter dimensão 2 e {tamanho_alfabeto} colunas.")
        return None

    for encoding in matriz_one_hot:
        indice_um = np.argmax(encoding)
        if np.sum(encoding) == 1 and 0 <= indice_um < tamanho_alfabeto:
            lista_de_letras.append(alfabeto[indice_um])
        else:
            print(f"Aviso: Codificação one-hot inválida encontrada: {encoding}. Ignorando.")
            lista_de_letras.append(None)  # Ou você pode escolher ignorar e não adicionar

    return [letra for letra in lista_de_letras if letra is not None] # Remove os None, se preferir

with open('dados\X.txt',"r") as arquivo:
    entradas = []
    for linha in arquivo:
        numeros = []
        valores = linha.strip().split(',')
        for valor in valores:
            if valor:
                numero = int(valor)
                numeros.append(numero)
        entradas.append(numeros)

with open('dados\Y_letra.txt',"r") as arquivo:
    saidas = []
    for linha in arquivo:
        saidas.append(linha.strip())

entradas = np.array(entradas).T
saidas_encoded = lista_de_letras_para_one_hot(saidas).T

# Separar os conjuntos de entrada
entradas_treino = entradas[:, :-260]
entradas_validacao = entradas[:, -260:-130]
entradas_teste = entradas[:, -130:]

# Separar os conjuntos de saída codificada
saidas_treino = saidas_encoded[:, :-260]
saidas_validacao = saidas_encoded[:, -260:-130]
saidas_teste = saidas_encoded[:, -130:]

parametros_treinados = treinamento_mlp_lento(entradas_treino,saidas_treino,120,64,26,0.2,100,True)

previsoes_treino = prever(entradas_teste, parametros_treinados)
previsoes_binarias = (previsoes_treino >= 0.5).astype(int)
acuracia_treino = np.mean(previsoes_binarias == saidas_teste.astype(int))
print(f"Acurácia nos dados de treinamento: {acuracia_treino * 100:.2f}%")