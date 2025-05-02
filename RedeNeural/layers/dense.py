import numpy as np

class DenseLayer:
    def __init__(self, entradas_dim, saidas_dim, ativacao=None):
        """
        Inicializa uma camada densa (totalmente conectada).

        Args:
            entradas_dim (int): Dimensão da entrada da camada.
            saidas_dim (int): Número de neurônios (dimensão da saída) da camada.
            ativacao (str, optional): Nome da função de ativação a ser usada ('relu', 'sigmoid', 'tanh', etc.). Defaults to None (ativação linear).
        """
        self.entradas_dim = entradas_dim
        self.saidas_dim = saidas_dim
        self.ativacao_nome = ativacao.lower() if ativacao else None
        # Inicialização dos pesos e biases
        self.weights = np.random.randn(entradas_dim, saidas_dim) * 0.01
        self.biases = np.zeros((1, saidas_dim)) 
        # Recuperação da função de ativação pelo nome
        self.ativacao_funcao = self._get_ativacao_funcao(self.ativacao_nome)
        # Daados a serem armazenados durante a execução
        self.entrada = None  
        self.somatorio_entrada = None 
        self.saida = None  # A
        self.grad_weights = None  # dW
        self.grad_biases = None  # db

    def _get_ativacao_funcao(self, name):
        # Funções de (ativacao.py) importadas aqui.
        if name == 'relu':
            from layers.activation import ReLU
            return ReLU()
        elif name == 'sigmoid':
            from layers.activation import Sigmoid
            return Sigmoid()
        elif name == 'tanh':
            from layers.activation import Tanh
            return Tanh()
        elif name is None:
            return lambda x: x  # Ativação linear
        else:
            raise ValueError(f"Função de ativação desconhecida: {name}")

    def __call__(self, x):
        """
        Realiza a passagem para frente através da camada densa.

        Args:
            x (numpy.ndarray): Tensor de entrada.

        Returns:
            numpy.ndarray: Saída da camada após a transformação linear e a função de ativação.
        """
        self.entrada = x
        self.somatorio_entrada = np.dot(x, self.weights) + self.biases
        self.saida = self.ativacao_funcao(self.somatorio_entrada)
        return self.saida

    def backward(self, grad_saida): 
        """
        Realiza a passagem para trás (backward pass) através da camada densa.

        Args:
            grad_saida (numpy.ndarray): Gradiente da perda em relação à saída desta camada (δ[l]).

        Returns:
            numpy.ndarray: Gradiente da perda em relação à entrada desta camada (δ[l−1]).
        """
        # 1. Calcula o gradiente após a derivada da função de ativação:
        #    δ[l] = δ[l+1] ∘ g′[l](Z[l])
        grad_atvacao = self.ativacao_funcao.backward(grad_saida)

        # 2. Armazena o gradiente de ativação como δ[l]
        delta_l = grad_atvacao

        # 3. Gradiente em relação aos pesos:
        #    dW[l] = δ[l] ⋅ A[l−1]ᵀ
        self.grad_weights = np.dot(self.entrada.T, delta_l)

        # 4. Gradiente em relação aos vieses (biases):
        #    db[l] = sum(δ[l], axis=0)
        self.grad_biases = np.sum(delta_l, axis=0, keepdims=True)

        # 5. Gradiente a ser propagado para a camada anterior:
        #    δ[l−1] = W[l] ⋅ δ[l]
        grad_entrada = np.dot(delta_l, self.weights.T)

        return grad_entrada


    def update(self, learning_rate):
        """
        Atualiza os pesos e biases da camada usando os gradientes calculados.

        Args:
            learning_rate (float): Taxa de aprendizado para a atualização.
        """
        self.weights -= learning_rate * self.grad_weights
        self.biases -= learning_rate * self.grad_biases