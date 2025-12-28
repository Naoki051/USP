import numpy as np
from concurrent.futures import ProcessPoolExecutor
from .mlp import MLP


def treinar_fold_worker(args):
    """
    Worker executado em processo separado.
    Recebe todos os argumentos em uma única tupla (pickle-safe).
    """
    (i, train_idx, val_idx, X, y,
     n_input, n_hidden, n_output, params) = args

    lr = params.get('lr', 0.1)
    lambda_ = params.get('lambda_', 0.01)
    epochs = params.get('epochs', 1000)
    patience = params.get('patience', 50)

    model = MLP(n_input, n_hidden, n_output, lr=lr, lambda_=lambda_)

    model.treinar(
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        epochs=epochs,
        patience=patience
    )

    y_pred = np.argmax(model.forward(X[val_idx]), axis=1)
    y_true = np.argmax(y[val_idx], axis=1)
    acc = np.mean(y_pred == y_true)

    return {
        'fold': i + 1,
        'score': acc,
        'train_loss': model.train_loss_history,
        'val_loss': model.val_loss_history
    }


def cross_validation_kfold(X, y, n_input, n_hidden, n_output, k=5, **params):
    indices = np.random.permutation(len(X))
    folds = np.array_split(indices, k)

    print(f"Iniciando Cross-Validation Paralela ({k} folds)...")

    tarefas = []
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.hstack([folds[j] for j in range(k) if j != i])

        tarefas.append((
            i, train_idx, val_idx,
            X, y,
            n_input, n_hidden, n_output,
            params
        ))

    scores = []
    historicos = []

    # IMPORTANTE: necessário para Windows
    with ProcessPoolExecutor(max_workers=k) as executor:
        resultados = list(executor.map(treinar_fold_worker, tarefas))

    # Ordena por número do fold
    resultados.sort(key=lambda x: x['fold'])

    for res in resultados:
        scores.append(res['score'])
        historicos.append({
            'train': res['train_loss'],
            'val': res['val_loss']
        })
        print(f"Fold {res['fold']} finalizado | Acurácia: {res['score']:.2%}")

    return scores, historicos
