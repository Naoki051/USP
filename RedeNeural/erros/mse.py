import numpy as np

class MeanSquaredError:
    def __call__(self, predicted, target):
        """
        Calcula o Erro Quadrático Médio (Mean Squared Error).

        Args:
            predicted (numpy.ndarray): As previsões da rede.
            target (numpy.ndarray): Os rótulos verdadeiros (targets).

        Returns:
            float: O valor médio do erro quadrático.
        """
        return np.mean((predicted - target)**2)

    def backward_gradient(self, predicted, target):
        """
        Calcula o gradiente do Erro Quadrático Médio em relação às previsões.

        Args:
            predicted (numpy.ndarray): As previsões da rede.
            target (numpy.ndarray): Os rótulos verdadeiros (targets).

        Returns:
            numpy.ndarray: O gradiente do MSE em relação às previsões.
        """
        return 2 * (predicted - target) / target.shape[0]