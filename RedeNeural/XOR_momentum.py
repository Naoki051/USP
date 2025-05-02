import numpy as np
from training import treinamento_mlp_com_momentum, plotar_custos_mlp, prever

if __name__ == '__main__':
    # Dados de treinamento de exemplo (porta XOR)
    X_treino = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2 features, 4 exemplos
    Y_treino = np.array([[0, 1, 1, 0]])  # Rótulos correspondentes (XOR gate)

    # Hiperparâmetros
    taxa_de_aprendizado = 0.1
    num_iteracoes = 1200
    num_entrada = X_treino.shape[0]  # 2
    num_oculta = 5
    num_saida = Y_treino.shape[0]  # 1
    beta_momentum = 0.1

    # Treina o modelo usando a função treinamento_mlp_com_momentum
    parametros_aprendidos, custos_historico = treinamento_mlp_com_momentum(
        X_treino, Y_treino, num_entrada, num_oculta, num_saida,
        taxa_de_aprendizado, num_iteracoes, beta=beta_momentum, imprimir_custo=True
    )

    # Plota os custos
    plotar_custos_mlp(custos_historico)

    # Avaliação do modelo treinado
    previsoes_treino = prever(X_treino, parametros_aprendidos)
    previsoes_binarias = (previsoes_treino >= 0.5).astype(int)
    print("\nPrevisões nos dados de treinamento:\n", previsoes_binarias)
    print("Rótulos verdadeiros nos dados de treinamento:\n", Y_treino.astype(int))

    acuracia_treino = np.mean(previsoes_binarias == Y_treino.astype(int))
    print(f"Acurácia nos dados de treinamento: {acuracia_treino * 100:.2f}%")