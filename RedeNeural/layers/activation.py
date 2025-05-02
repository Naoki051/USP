import numpy as np

class Activation:
    """Classe base para funções de ativação."""
    def __call__(self, x):
        """Aplica a função de ativação ao input."""
        raise NotImplementedError("Subclasses devem implementar o método forward.")

    def backward(self, grad_output):
        """Calcula o gradiente da função de ativação."""
        raise NotImplementedError("Subclasses devem implementar o método backward.")

class ReLU(Activation):
    """Função de ativação ReLU (Rectified Linear Unit)."""
    def __call__(self, x):
        self.output = np.maximum(0, x)
        return self.output

    def backward(self, grad_output):
        # δ[L] =  δ[l+1]   ∘    g[L]′(Z[L])
        return grad_output * (self.output > 0)

class Sigmoid(Activation):
    """Função de ativação Sigmoid."""
    def __call__(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self, grad_output):
        # δ[L] =  δ[l+1]   ∘         g[L]′(Z[L])
        return grad_output * self.output * (1 - self.output)

class Tanh(Activation):
    """Função de ativação Tangente Hiperbólica."""
    def __call__(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        # δ[L] =  δ[l+1]   ∘    g[L]′(Z[L])
        return grad_output * (1 - self.output**2)

class Softmax(Activation):
    """Função de ativação Softmax."""
    def __call__(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        n = self.output.shape[-1]
        return np.array([np.dot((np.diagflat(out) - np.outer(out, out)), g)
                         for out, g in zip(self.output, grad_output)])

class Linear(Activation):
    """Função de ativação Linear (identidade)."""
    def __call__(self, x):
        self.output = x
        return self.output

    def backward(self, grad_output):
        return grad_output * 1

# Você pode adicionar mais funções de ativação conforme necessário (LeakyReLU, ELU, etc.)