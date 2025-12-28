import os
import pickle
import numpy as np

from src.mlp import MLP
from src.data_utils import preparar_dados
from src.validation import cross_validation_kfold
from src.plots import plotar_kfold_completo


def main():
    # ============================
    # CONFIGURAÇÕES GERAIS
    # ============================
    DATA_PATH = 'data/penguins_lter.csv'
    MODEL_PATH = 'data/mlp_pinguins.pkl'

    N_HIDDEN = 12
    LR = 0.15
    LAMBDA = 0.01
    K_FOLDS = 5

    EPOCHS_CV = 50000     # menor que o treino final
    EPOCHS_FINAL = 1500
    PATIENCE = 5

    # ============================
    # PREPARAÇÃO DO AMBIENTE
    # ============================
    os.makedirs('data', exist_ok=True)

    # ============================
    # 1. PREPARAÇÃO DOS DADOS
    # ============================
    print("\n=== 1. Carregando e Preparando Dados ===")
    X, y, x_min, x_max, classes = preparar_dados(DATA_PATH)

    # ============================
    # 2. CROSS-VALIDATION PARALELA
    # ============================
    print("\n=== 2. Cross-Validation (K-Fold Paralelo) ===")

    scores, historicos = cross_validation_kfold(
        X, y,
        n_input=X.shape[1],
        n_hidden=N_HIDDEN,
        n_output=y.shape[1],
        k=K_FOLDS,
        lr=LR,
        lambda_=LAMBDA,
        epochs=EPOCHS_CV,
        patience=PATIENCE
    )

    media = np.mean(scores)
    desvio = np.std(scores)

    print(f"\nAcurácia Média Final: {media:.2%} ± {desvio:.2%}")

    # ============================
    # 3. VISUALIZAÇÃO
    # ============================
    plotar_kfold_completo(scores, historicos)

    # ============================
    # 4. TREINAMENTO FINAL
    # ============================
    print("\n=== 3. Treinando Modelo Final ===")

    modelo = MLP(
        n_input=X.shape[1],
        n_hidden=N_HIDDEN,
        n_output=y.shape[1],
        lr=LR,
        lambda_=LAMBDA
    )

    modelo.X_min = x_min
    modelo.X_max = x_max
    modelo.classes = classes

    # Usamos X e y como validação apenas para early stopping
    modelo.treinar(
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        epochs=EPOCHS_FINAL,
        patience=PATIENCE
    )

    print(f"Treinamento final concluído em {modelo.tempo_execucao:.2f}s")

    # ============================
    # 5. PERSISTÊNCIA DO MODELO
    # ============================
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(modelo, f)

    print(f"\nModelo salvo com sucesso em: {MODEL_PATH}")

    # ============================
    # 6. TESTE DE INFERÊNCIA
    # ============================
    print("\n=== 4. Teste de Inferência ===")

    amostra = np.array([50.0, 18.0, 210.0, 4500.0])
    especie, confianca = modelo.prever_especie(amostra)

    print(f"Entrada: {amostra}")
    print(f"Predição: {especie} | Confiança: {confianca:.2%}")


# ============================
# OBRIGATÓRIO PARA MULTIPROCESSING
# ============================
if __name__ == "__main__":
    main()
