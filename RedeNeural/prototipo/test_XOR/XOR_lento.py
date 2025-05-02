import numpy as np
from training import treinamento_mlp_lento, prever

# Dados de treinamento de exemplo (porta XOR)
X_treino = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T  # 2 features, 4 exemplos
Y_treino = np.array([[0, 1, 1, 0]])  # Rótulos correspondentes (XOR gate)

# Hiperparâmetros
taxa_de_aprendizado = 0.1
num_iteracoes = 5000
num_entrada = X_treino.shape[0]  # 2
num_oculta = 12
num_saida = Y_treino.shape[0]  # 1

# Treina o modelo usando a função treinamento_mlp e obtém os custos
parametros_aprendidos = treinamento_mlp_lento(X_treino, Y_treino, num_entrada, num_oculta, num_saida, taxa_de_aprendizado, num_iteracoes, imprimir_custo=True)
# Avaliação do modelo treinado
previsoes_treino = prever(X_treino, parametros_aprendidos)
previsoes_binarias = (previsoes_treino >= 0.5).astype(int)
print("\nPrevisões nos dados de treinamento:\n", previsoes_treino)
print("Rótulos verdadeiros nos dados de treinamento:\n", Y_treino.astype(int))
previsoes_binarias = (previsoes_treino >= 0.5).astype(int)
acuracia_treino = np.mean(previsoes_binarias == Y_treino.astype(int))
print(f"Acurácia nos dados de treinamento: {acuracia_treino * 100:.2f}%")