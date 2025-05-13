import numpy as np
import string
from models.mlp import MLP
import matplotlib.pyplot as plt

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

# Carregar dados de entrada do arquivo X.txt
try:
    with open('data/X.txt',"r") as arquivo:
        entradas = []
        for linha in arquivo:
            numeros = []
            valores = linha.strip().split(',')
            for valor in valores:
                if valor:
                    numero = int(valor)
                    numeros.append(numero)
            entradas.append(numeros)
    entradas = np.array(entradas)
    print(f"Dados de entrada carregados com sucesso. Shape: {entradas.shape}")
except FileNotFoundError:
    print("Erro: O arquivo 'data/X.txt' não foi encontrado.")
    entradas = None
except Exception as e:
    print(f"Ocorreu um erro ao carregar 'data/X.txt': {e}")
    entradas = None

# Carregar rótulos de saída do arquivo Y_letra.txt e codificar para one-hot
try:
    with open('data/Y_letra.txt',"r") as arquivo:
        saidas = []
        for linha in arquivo:
            saidas.append(linha.strip())
    saidas_encoded = lista_de_letras_para_one_hot(saidas)
    print(f"Rótulos de saída carregados e codificados com sucesso. Shape: {saidas_encoded.shape}")
except FileNotFoundError:
    print("Erro: O arquivo 'data/Y_letra.txt' não foi encontrado.")
    saidas_encoded = None
except Exception as e:
    print(f"Ocorreu um erro ao carregar 'data/Y_letra.txt': {e}")
    saidas_encoded = None

# Separar os dados se foram carregados corretamente
if entradas is not None and saidas_encoded is not None and len(entradas) == len(saidas_encoded):
    num_total_samples = len(entradas)
    num_test = 130
    num_val = 130

    if num_total_samples >= num_test + num_val:
        # Separar os dados de teste (os últimos 130)
        X_test = entradas[-num_test:]
        Y_test = saidas_encoded[-num_test:]

        # Separar os dados de validação (os 130 anteriores aos de teste)
        X_val = entradas[-(num_test + num_val):-num_test]
        y_val = saidas_encoded[-(num_test + num_val):-num_test]

        # O restante dos dados será para treinamento
        X_train = entradas[:-(num_test + num_val)]
        y_train = saidas_encoded[:-(num_test + num_val)]

        print(f"\nNúmero de amostras para treinamento: {len(X_train)}")
        print(f"Número de amostras para validação: {len(X_val)}")
        print(f"Número de amostras para teste: {len(X_test)}")

        # 2. Inicialização do Modelo
        # O número de neurônios na camada de entrada deve ser a dimensão de cada amostra em X_train (120)
        # O número de neurônios na camada de saída deve ser o tamanho da codificação one-hot (26 para o alfabeto)
        model = MLP(entradas_dim=120, oculta_dim=[128], saidas_dim=26, ativacao='tanh', saidas_ativacao='sigmoid')

        # 3. Hiperparâmetros
        taxa_aprendizado = 0.01
        epocas = 50
        # Vai passar cada entrada individualmente e atualizar pesos
        lote_tamanho = 1 

        # Listas para armazenar as perdas
        train_losses = []
        val_losses = []

        # Função para criar lotes de dados
        def create_lotes(X, y, lote_tamanho):
            m = X.shape[0]
            lotes = []
            for i in range(0, m, lote_tamanho):
                X_lote = X[i:i + lote_tamanho]
                y_lote = y[i:i + lote_tamanho]
                lotes.append((X_lote, y_lote))
            return lotes

        # 4. Loop de Treinamento
        for epoch in range(epocas):
            indices = np.arange(X_train.shape[0])
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            train_lotes = create_lotes(X_train, y_train, lote_tamanho)
            total_train_loss = 0

            for lote_inputs, lote_targets in train_lotes:
                train_loss = model.train_step(lote_inputs, lote_targets, taxa_aprendizado)
                total_train_loss += train_loss

            avg_train_loss = total_train_loss / len(train_lotes)
            train_losses.append(avg_train_loss)

            # 5. Validação
            val_loss = model.evaluate(X_val, y_val)
            val_losses.append(val_loss)

            # 6. Impressão de Resultados
            print(f"Época {epoch+1}/{epocas}, Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        print("Treinamento concluído!")
        # Cálculo da acurácia
        output = model.forward(X_test)
        predicoes = np.argmax(output, axis=1)
        target_output = Y_test
        verdadeiros = np.argmax(target_output, axis=1)
        acuracia = np.mean(predicoes == verdadeiros)
        print(f"\nDesempenho no conjunto de teste — Accuracy: {acuracia:.4f}")

        # 7. Plot das Curvas de Perda
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epocas + 1), train_losses, label='Training Loss')
        plt.plot(range(1, epocas + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Curves')
        plt.legend()
        plt.grid(True)
        plt.show()

    else:
        print(f"\nErro: Número total de amostras ({num_total_samples}) é insuficiente para separar {num_test} para teste e {num_val} para validação.")
elif entradas is None or saidas_encoded is None:
    print("\nErro ao carregar os dados. O treinamento não pôde ser realizado.")
elif len(entradas) != len(saidas_encoded):
    print(f"\nErro: O número de amostras de entrada ({len(entradas)}) não corresponde ao número de rótulos ({len(saidas_encoded)}). O treinamento não pôde ser realizado.")