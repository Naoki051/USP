import numpy as np
from mlp import novo_modelo,tanh,sigmoid,train_step,feedforward

# Inicializando a rede MLP

modelo = novo_modelo(
    entrada_dim=2,
    ocultas_dim=4,
    saida_dim=1,
    ativacao_oculta=tanh,
    ativacao_saida=sigmoid
)

# Dados de entrada (XOR)
entradas = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
saidas_esperadas = np.array([
    [0],
    [1],
    [1],
    [0]
])

# Treinamento
losses = []
for epoca in range(150):
    loss = train_step(entradas, saidas_esperadas, modelo, taxa_aprendizado=0.9)
    losses.append(loss)
    if epoca%100==0:
        print(f"{epoca}/500 Loss: {loss:.4f}")
# Saídas finais
predicoes = []
entrada = entradas
for i in range(len(entrada)):
    x = entrada[i].reshape(1, -1)
    for camada in modelo:
        x = feedforward(camada, x)
    predicoes.append(x)

predicoes = np.vstack(predicoes)
losses[-1], predicoes.squeeze()

import matplotlib.pyplot as plt

# Plot da curva de perda
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title('Curva de Perda durante o Treinamento')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.grid(True)
plt.show()

# Saídas finais (seu código para obter as predições finais)
predicoes = []
entrada = entradas
for i in range(len(entrada)):
    x = entrada[i].reshape(1, -1)
    for camada in modelo:
        x = feedforward(camada, x)
    predicoes.append(x)

predicoes = np.vstack(predicoes)

print("Estrutura do Modelo:")
for i, camada in enumerate(modelo):
    print(f"--- Camada {i+1} ---")
    for chave, valor in camada.items():
        if isinstance(valor, np.ndarray):
            print(f"{chave.capitalize()}:\n{valor}")
        elif callable(valor):
            print(f"{chave.capitalize()}: {valor.__name__}")
        else:
            print(f"{chave.capitalize()}: {valor}")

print(f"Perda final: {losses[-1]}")
print(f"Predições:\n{predicoes.squeeze()}")