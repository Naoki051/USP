import numpy as np
from layers.dense import DenseLayer
from erros.mse import MeanSquaredError

class MLP:
    def __init__(self, entradas_dim, oculta_dim, saidas_dim, ativacao='relu', saidas_ativacao='sigmoid'):
        """
        Inicializa a Rede Neural - Multi Layer Perceptron (MLP).

        Args:
            entradas_dim (int): Dimensão da camada de entrada.
            oculta_dim (list of int): Lista com as dimensões de cada camada oculta.
            saidas_dim (int): Dimensão da camada de saída.
            ativacao (str): Nome da função de ativação a ser usada nas camadas ocultas ('relu', 'sigmoid', 'tanh', etc.).
            saidas_ativacao (str): Nome da função de ativação a ser usada na camada de saída.
        """
        # Armazenando parâmetros em variáveis locais
        self.entradas_dim = entradas_dim
        self.oculta_dim = oculta_dim
        self.saidas_dim = saidas_dim
        self.ativacao_nome = ativacao.lower()
        self.saidas_ativacao_nome = saidas_ativacao.lower()
        self.loss_function = MeanSquaredError() # Instancia a função de perda

        # Lista para armazenar as camadas
        self.layers = []  

        # 1. Primeira camada (entrada para a primeira camada oculta)
        if oculta_dim:
            self.layers.append(DenseLayer(entradas_dim, oculta_dim[0], ativacao=self.ativacao_nome))
            prev_dim = oculta_dim[0]
            # 2. Camadas ocultas intermediárias
            for i in range(1, len(oculta_dim)):
                self.layers.append(DenseLayer(prev_dim, oculta_dim[i], ativacao=self.ativacao_nome))
                prev_dim = oculta_dim[i]
            # 3. Camada de saída
            self.layers.append(DenseLayer(prev_dim, saidas_dim, ativacao=self.saidas_ativacao_nome))
        else:
            # Caso não haja camadas ocultas, apenas uma camada de saída diretamente da entrada
            self.layers.append(DenseLayer(entradas_dim, saidas_dim, ativacao=self.saidas_ativacao_nome))


    def forward(self, x):
        """
        Realiza a passagem para frente (forward pass) através da rede.

        Args:
            x (numpy.ndarray): Tensor de entrada.

        Returns:
            numpy.ndarray: Saída da rede.
        """
        output = x
        for layer in self.layers:
            output = layer(output)  # Supondo que cada objeto 'layer' tenha um método '__call__' que implementa o forward
        return output

    def backward(self, grad_output):
        """
        Realiza a passagem para trás (backward pass) através da rede.

        Args:
            grad_output (numpy.ndarray): Gradiente da perda em relação à saída da rede.
        """
        grad_input = grad_output
        # Itera sobre as camadas na ordem reversa
        for layer in reversed(self.layers):
            grad_input = layer.backward(grad_input)
        # O gradiente final (grad_input) é o gradiente da perda em relação à entrada da rede
        return grad_input

    def update_weights(self, learning_rate):
        """
        Atualiza os pesos de todas as camadas da rede.

        Args:
            learning_rate (float): Taxa de aprendizado para a atualização.
        """
        for layer in self.layers:
            if hasattr(layer, 'update'):
                layer.update(learning_rate)

    def train_step(self, input_data, target_output, learning_rate):
        """
        Realiza uma única etapa de treinamento: forward, backward e atualização de pesos.

        Args:
            input_data (numpy.ndarray): Lote de dados de entrada.
            target_output (numpy.ndarray): Lote de rótulos verdadeiros (targets).
            learning_rate (float): Taxa de aprendizado para a atualização.

        Returns:
            float: O valor da perda (loss) para este lote.
        """
        
        
        # 1. Forward pass
        output = self.forward(input_data)

        # 2. Calcular a perda (Erro quadrático médio)
        loss = self.loss_function(output, target_output)

        # 3. Calcular o gradiente da perda em relação à saída da rede
        grad_output = self.loss_function.backward_gradient(output, target_output)

        # 4. Backward pass
        self.backward(grad_output)

        # 5. Atualizar os pesos
        self.update_weights(learning_rate)

        return loss

    def evaluate(self, input_data, target_output):
        """
        Realiza a passagem para frente (forward pass) e calcula a perda
        para um conjunto de dados de validação ou teste.

        Args:
            input_data (numpy.ndarray): Lote de dados de entrada.
            target_output (numpy.ndarray): Lote de rótulos verdadeiros (targets).

        Returns:
            float: O valor da perda (loss) para este lote.
        """
        output = self.forward(input_data)
        loss = self.loss_function(output, target_output)
        return loss