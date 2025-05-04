## ***Fundamentação Teórica e Derivação Matemática da Retropropagação em Redes Neurais MLP com Atualização Baseada no Erro Quadrático Médio***

---

### **1. Objetivo**

Este relatório tem como foco a explicação matemática dos cálculos realizados pelo modelo de rede neural MLP (Perceptron Multicamadas) implementado no arquivo `mlp.py`. O destaque será para as **derivações e fórmulas utilizadas na retropropagação do erro (backpropagation)**, que permite o aprendizado da rede por meio da atualização de pesos.

---

### **2. Funções de Ativação e Derivadas**

A retropropagação utiliza derivadas das funções de ativação. Neste projeto, foram utilizadas:

* **Função `tanh(x)`** na camada oculta:

$$
  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \quad \text{com derivada:} \quad \tanh'(x) = 1 - \tanh^2(x)
$$

* **Função `sigmoid(x)`** na saída:

$$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \quad \text{com derivada:} \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
$$

Essas derivadas são usadas no cálculo dos gradientes para atualizar os pesos.

---

### **3. Propagação Direta (Forward Propagation)**

Durante a fase de propagação direta, a rede calcula:

* **Ativação da camada oculta:**

$$
  z^{(1)} = X \cdot W^{(1)} + b^{(1)} \quad (\text{soma ponderada})
$$

$$
  a^{(1)} = \tanh(z^{(1)}) \quad (\text{aplicação da ativação})
$$

* **Ativação da camada de saída:**

$$
  z^{(2)} = a^{(1)} \cdot W^{(2)} + b^{(2)}
$$

$$
  a^{(2)} = \sigma(z^{(2)})
$$

Isso é implementado no método `forward()` da classe `MLP`.

---

### **4. Função de Custo (Loss Function)**

A função de perda utilizada foi o **Erro Quadrático Médio (MSE)**:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Na implementação:

```python
loss = np.mean((y - output) ** 2)
```

---

### **5. Retropropagação: Derivadas e Atualização dos Pesos**

A retropropagação consiste em calcular os **gradientes do erro em relação aos pesos e biases de cada camada**, com base na regra da cadeia do cálculo diferencial.

#### 5.1. Erro na camada de saída (\$\delta^{\[L]}\$):

$$
\delta^{[L]} = \frac{\partial L}{\partial a^{[L]}} \odot g'^{[L]}(z^{[L]})
$$

* Para MSE: \$\frac{\partial L}{\partial a^{\[L]}} = 2(a^{\[L]} - y)\$
* Para Sigmoid: \$g'^{\[L]}(z^{\[L]}) = a^{\[L]}(1 - a^{\[L]})\$

> Implementado nos métodos `backward_gradient` da classe `MeanSquaredError` e `backward` da classe `Sigmoid`.

#### 5.2. Gradiente dos pesos da camada de saída (\$W^{\[L]}\$):

$$
\frac{\partial L}{\partial W^{[L]}} = (a^{[L-1]})^T \cdot \delta^{[L]}
$$

> Implementado como:

```python
self.grad_weights = np.dot(self.entrada.T, grad_ativacao)
```

#### 5.3. Gradiente dos biases da camada de saída (\$b^{\[L]}\$):

$$
\frac{\partial L}{\partial b^{[L]}} = \sum \delta^{[L]}, \text{ao longo das amostras}
$$

> Implementado como:

```python
self.grad_biases = np.sum(grad_ativacao, axis=0, keepdims=True)
```

#### 5.4. Erro nas camadas ocultas (\$\delta^{\[l]}\$):

$$
\delta^{[l]} = (W^{[l+1]} \cdot \delta^{[l+1]}) \odot g'^{[l]}(z^{[l]})
$$

> Implementado como:

```python
grad_entrada = np.dot(grad_ativacao, self.weights.T)
```

Multiplicado pela derivada da ativação na próxima iteração do `backward` da `MLP`.

#### 5.5 e 5.6. Gradientes dos pesos e biases das camadas ocultas:

Mesmas fórmulas das camadas de saída, aplicadas às camadas intermediárias.

---

### **6. Atualização dos Pesos (Gradient Descent)**

Com os gradientes computados, aplica-se o passo do gradiente descendente:

$$
W := W - \eta \cdot \frac{\partial L}{\partial W}
\quad
b := b - \eta \cdot \frac{\partial L}{\partial b}
$$

> Implementado no método `update()` da classe `DenseLayer`:

```python
self.weights -= learning_rate * self.grad_weights
self.biases -= learning_rate * self.grad_biases
```

---

### **7. Ciclo de Treinamento (`train_step`)**

O método `train_step` realiza uma iteração completa de aprendizado:

1. **Previsão:**

   ```python
   output = self.forward(input_data)
   ```

2. **Erro:**

   ```python
   loss = self.loss_function(output, target_output)
   ```

3. **Gradiente do erro:**

   ```python
   grad_output = self.loss_function.backward_gradient(output, target_output)
   ```

4. **Retropropagação:**

   ```python
   self.backward(grad_output)
   ```

5. **Atualização dos pesos:**

   ```python
   self.update_weights(learning_rate)
   ```

Retorna o `loss` para monitoramento.

---

### **8. Avaliação (`evaluate`)**

Esse método mede o desempenho da rede em dados de teste (sem ajuste de pesos):

1. **Previsão:**

   ```python
   output = self.forward(input_data)
   ```

2. **Cálculo da perda:**

   ```python
   loss = self.loss_function(output, target_output)
   ```

---

### **9. Conclusão**

A implementação da rede MLP descrita neste relatório apresenta uma estrutura clara e didática, permitindo o entendimento detalhado de cada etapa do aprendizado. A separação entre funções de ativação, camadas densas, função de perda e ciclos de treino reflete boas práticas de engenharia de software e facilita futuras modificações.

A explicação matemática fornecida mostra como conceitos fundamentais de cálculo diferencial, como a regra da cadeia, são utilizados para ajustar os pesos da rede de forma eficiente. A integração entre teoria e prática está bem representada no código `mlp.py`, consolidando o entendimento de como redes neurais realmente "aprendem".

Esse modelo serve como base sólida para experimentações futuras com arquiteturas mais complexas, outras funções de ativação ou métodos de otimização mais avançados, como Adam ou RMSprop. Além disso, é uma excelente ferramenta de aprendizado para quem está dando os primeiros passos no campo de aprendizado de máquina.

---
