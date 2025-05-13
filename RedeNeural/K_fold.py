import numpy as np
import string, json, os
import matplotlib.pyplot as plt
from multiprocessing import Process, Manager
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
            return np.array([[int(val) for val in linha.strip().split(',') if val] for linha in f])
    except Exception as e:
        print(f"Erro ao carregar '{caminho}': {e}")
        return None


def carregar_saidas(caminho):
    try:
        with open(caminho, "r") as f:
            letras = [linha.strip() for linha in f]
        return lista_de_letras_para_one_hot(letras)
    except Exception as e:
        print(f"Erro ao carregar '{caminho}': {e}")
        return None


def create_lotes(X, y, tamanho_lote):
    return [(X[i:i + tamanho_lote], y[i:i + tamanho_lote]) for i in range(0, X.shape[0], tamanho_lote)]


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

        if epoca % 10 == 0:
            print(f"[{os.getpid()}] Época {epoca+1} - Train: {avg_train_loss:.4f}, Val: {val_loss:.4f}")
    return train_losses, val_losses


def avaliar_modelo(model, X_test, y_test):
    output = model.forward(X_test)
    pred = np.argmax(output, axis=1)
    real = np.argmax(y_test, axis=1)
    return np.mean(pred == real)


def salvar_modelo(model, caminho):
    parametros = []
    for i, camada in enumerate(model.layers):
        parametros.append({
            f'camada_{i+1}_pesos': camada.weights.tolist(),
            f'camada_{i+1}_bias': camada.biases.tolist()
        })
    with open(caminho, 'w') as f:
        json.dump(parametros, f, indent=4)


def salvar_perdas(train_losses, val_losses, caminho):
    with open(caminho, 'w') as f:
        json.dump({
            "train_losses": train_losses,
            "val_losses": val_losses
        }, f, indent=4)


def treinar_fold(i, entradas, saidas, X_test, Y_test, epocas, taxa, lote, acuracias_compartilhadas):
    fold_size = len(entradas) // 13
    ini, fim = i * fold_size, (i + 1) * fold_size
    val_idx = np.arange(ini, fim)
    train_idx = np.delete(np.arange(len(entradas)), val_idx)

    X_train, y_train = entradas[train_idx], saidas[train_idx]
    X_val, y_val = entradas[val_idx], saidas[val_idx]

    model = MLP(entradas_dim=120, oculta_dim=[128], saidas_dim=26, ativacao='tanh', saidas_ativacao='sigmoid')
    print(f"\n[{os.getpid()}] Treinando Fold {i+1}/13...")

    train_losses, val_losses = treinar_modelo(model, X_train, y_train, X_val, y_val, epocas, taxa, lote)

    os.makedirs('pesos', exist_ok=True)
    salvar_modelo(model, f'pesos/modelo_fold_{i+1}.json')
    salvar_perdas(train_losses, val_losses, f'pesos/perdas_fold_{i+1}.json')

    acc = avaliar_modelo(model, X_test, Y_test)
    acuracias_compartilhadas[i] = acc
    print(f"[{os.getpid()}] Acurácia do Fold {i+1}: {acc:.4f}")


def main():
    entradas = carregar_entradas('data/X.txt')
    saidas = carregar_saidas('data/Y_letra.txt')
    if entradas is None or saidas is None:
        return

    num_test = 130
    epocas = 50
    taxa = 0.01
    lote = 1
    X_test = entradas[-num_test:]
    Y_test = saidas[-num_test:]
    entradas = entradas[:-num_test]
    saidas = saidas[:-num_test]

    with Manager() as manager:
        acuracias = manager.list([0.0] * 13)
        processos = []

        for i in range(13):
            p = Process(
                target=treinar_fold,
                args=(i, entradas, saidas, X_test, Y_test, epocas, taxa, lote, acuracias)
            )
            processos.append(p)
            p.start()

        for p in processos:
            p.join()

        # Resultados
        acuracias_finais = list(acuracias)
        print("\nResumo das Acurácias:")
        for i, acc in enumerate(acuracias_finais):
            print(f"Fold {i+1}: {acc:.4f}")
        print(f"Média: {np.mean(acuracias_finais):.4f}")
        print(f"Desvio Padrão: {np.std(acuracias_finais):.4f}")


if __name__ == "__main__":
    main()
