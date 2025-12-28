import numpy as np
import time


class MLP:
    def __init__(self, n_input, n_hidden, n_output, lr=0.1, lambda_=0.01):
        self.lr = lr
        self.lambda_ = lambda_

        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(1. / n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(1. / n_hidden)
        self.b2 = np.zeros((1, n_output))

        self.X_min = None
        self.X_max = None
        self.classes = None

        self.train_loss_history = []
        self.val_loss_history = []
        self.tempo_execucao = 0

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def _sigmoid_derivative(self, a):
        return a * (1 - a)

    def _softmax(self, z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def forward(self, X):
        self.A1 = self._sigmoid(X @ self.W1 + self.b1)
        self.A2 = self._softmax(self.A1 @ self.W2 + self.b2)
        return self.A2

    def backward(self, X, y):
        m = y.shape[0]

        dZ2 = self.A2 - y
        dW2 = (self.A1.T @ dZ2) / m + (self.lambda_ / m) * self.W2
        db2 = dZ2.mean(axis=0, keepdims=True)

        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * self._sigmoid_derivative(self.A1)
        dW1 = (X.T @ dZ1) / m + (self.lambda_ / m) * self.W1
        db1 = dZ1.mean(axis=0, keepdims=True)

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def calcular_loss(self, X, y):
        A2 = self.forward(X)
        m = y.shape[0]
        loss = -np.sum(y * np.log(A2 + 1e-15)) / m
        l2 = (self.lambda_ / (2 * m)) * (np.sum(self.W1**2) + np.sum(self.W2**2))
        return loss + l2

    def treinar(self, X_train, y_train, X_val, y_val, epochs=2000, patience=50):
        inicio = time.time()
        melhor_loss = float('inf')
        sem_melhora = 0

        # Limpa histórico caso o modelo seja reutilizado
        self.train_loss_history = []
        self.val_loss_history = []

        for ep in range(epochs):
            # ======================
            # TREINAMENTO
            # ======================
            self.forward(X_train)
            self.backward(X_train, y_train)

            # ======================
            # MONITORAMENTO
            # ======================
            tl = self.calcular_loss(X_train, y_train)
            vl = self.calcular_loss(X_val, y_val)

            self.train_loss_history.append(tl)
            self.val_loss_history.append(vl)

            # ======================
            # EARLY STOPPING
            # ======================
            if vl < melhor_loss - 1e-6:
                melhor_loss = vl
                self.best_W1 = self.W1.copy()
                self.best_b1 = self.b1.copy()
                self.best_W2 = self.W2.copy()
                self.best_b2 = self.b2.copy()
                sem_melhora = 0
            else:
                sem_melhora += 1

            if sem_melhora >= patience:
                print(f"🛑 Early stopping na época {ep}")
                self.W1, self.b1 = self.best_W1, self.best_b1
                self.W2, self.b2 = self.best_W2, self.best_b2
                break

        self.tempo_execucao = time.time() - inicio

    def prever_especie(self, X_bruto):
        X_norm = (X_bruto - self.X_min) / (self.X_max - self.X_min)
        prob = self.forward(X_norm.reshape(1, -1))
        idx = np.argmax(prob)
        return self.classes[idx], prob[0, idx]
