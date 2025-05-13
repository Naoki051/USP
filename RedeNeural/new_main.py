import numpy as np
import string, json, os
import matplotlib.pyplot as plt
from models.mlp import MLP


def lista_de_letras_para_one_hot(lista_de_letras):
    alfabeto = string.ascii_uppercase
    tamanho = len(alfabeto)
    matriz = []

    for letra in lista_de_letras:
        if len(letra) == 1 and letra in alfabeto:
            vetor = np.zeros(tamanho, dtype=int)
            vetor[alfabeto.index(letra)] = 1
            matriz.append(vetor)
        else:
            print(f"Aviso: '{letra}' inválida, ignorada.")
    return np.array(matriz)


def carregar_entradas(caminho):
    try:
        with open(caminho, "r") as f:
            return np.array([
                [int(val) for val in linha.strip().split(',') if val]
                for linha in f
            ])
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho}' não encontrado.")
    except Exception as e:
        print(f"Erro ao carregar '{caminho}': {e}")
    return None


def carregar_saidas(caminho):
    try:
        with open(caminho, "r") as f:
            letras = [linha.strip() for linha in f]
        return lista_de_letras_para_one_hot(letras)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho}' não encontrado.")
    except Exception as e:
        print(f"Erro ao carregar '{caminho}': {e}")
    return None


def create_lotes(X, y, tamanho_lote):
    return [
        (X[i:i + tamanho_lote], y[i:i + tamanho_lote])
        for i in range(0, X.shape[0], tamanho_lote)
    ]


def treinar_modelo(model, X_train, y_train, X_val, y_val, epocas, taxa, tamanho_lote):
    train_losses, val_losses = [], []
    for epoca in range(epocas):
        indices = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[indices], y_train[indices]
        lotes = create_lotes(X_train, y_train, tamanho_lote)

        total_loss = sum(model.train_step(x, y, taxa) for x, y in lotes)
        avg_train_loss = total_loss / len(lotes)
        val_loss = model.evaluate(X_val, y_val)

        train_losses.append(avg_train_loss)
        val_losses.append(val_loss)
        if(epoca%10==0):
            print(f"Época {epoca+1}/{epocas} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
    return train_losses, val_losses


def avaliar_modelo(model, X_test, y_test):
    output = model.forward(X_test)
    pred = np.argmax(output, axis=1)
    real = np.argmax(y_test, axis=1)
    return np.mean(pred == real)

def salvar_modelo(model, caminho):
    """Salva os pesos e biases do modelo em um arquivo JSON."""
    parametros = []
    try:
        for i, camada in enumerate(model.layers):
            w = camada.weights.T[-1]
            b = camada.biases.T[-1]
            parametros.append({
                f'camada_{i+1}_pesos': w.tolist(),
                f'camada_{i+1}_bias': b.tolist()
            })

        with open(caminho, 'w') as arquivo_json:
            json.dump(parametros, arquivo_json, indent=4)
        print(f"Pesos e biases salvos em formato JSON em: {caminho}")
    except Exception as e:
        print(f"Erro ao salvar o modelo em JSON: {e}")



def main():
    entradas = carregar_entradas('data/X.txt')
    saidas = carregar_saidas('data/Y_letra.txt')
    if entradas is None or saidas is None:
        return

    num_total = len(entradas)
    num_test = 130
    num_folds = 13
    epocas = 50
    taxa = 0.01
    lote = 1

    X_test = entradas[-num_test:]
    Y_test = saidas[-num_test:]

    acuracias = []
    fold_size = num_total // num_folds

    for i in range(num_folds):
        ini, fim = i * fold_size, (i + 1) * fold_size
        val_idx = np.arange(ini, fim)
        train_idx = np.delete(np.arange(num_total), val_idx)

        X_train, y_train = entradas[train_idx], saidas[train_idx]
        X_val, y_val = entradas[val_idx], saidas[val_idx]

        model = MLP(entradas_dim=120, oculta_dim=[128], saidas_dim=26, ativacao='tanh', saidas_ativacao='sigmoid')
        print(f"\n--- Treinando Fold {i+1}/{num_folds} ---")
        train_losses, val_losses = treinar_modelo(model, X_train, y_train, X_val, y_val, epocas, taxa, lote)
        # armazenar pesos e bias após treinar
        nome_arquivo_fold = f'pesos/modelo_fold_{i+1}.json'
        os.makedirs('pesos',exist_ok=True)
        salvar_modelo(model, nome_arquivo_fold)
        acc = avaliar_modelo(model, X_test, Y_test)
        print(f"Acurácia no teste (Fold {i+1}): {acc:.4f}")
        acuracias.append(acc)

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Treino')
    plt.plot(val_losses, label='Validação')
    plt.title(f'Curva de Perda - Fold {i+1}')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.show()

    print("\nResumo das Acurácias:")
    print("Acurácias por fold:", [f"{acuracia:.4f}" for acuracia in acuracias])
    print(f"Média: {np.mean(acuracias):.4f}, Desvio Padrão: {np.std(acuracias):.4f}")


if __name__ == "__main__":
    main()
