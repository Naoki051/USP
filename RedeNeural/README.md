### **1. Objetivo**

Este relatório tem como foco a explicação matemática dos cálculos realizados pelo modelo de rede neural MLP (Perceptron Multicamadas) implementado no arquivo `mlp.py`. O destaque será para as **derivações e fórmulas utilizadas na retropropagação do erro (backpropagation)**, que permite o aprendizado da rede via atualização de pesos.

---

### **2. Funções Ativação e Derivadas**

A retropropagação utiliza derivadas das funções de ativação. Neste projeto, foram utilizadas:

- **Função `tanh(x)`** na camada oculta:

  $$
  \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
  \quad \text{com derivada:} \quad \tanh'(x) = 1 - \tanh^2(x)
  $$

- **Função `sigmoid(x)`** na saída:

  $$
  \sigma(x) = \frac{1}{1 + e^{-x}}
  \quad \text{com derivada:} \quad \sigma'(x) = \sigma(x)(1 - \sigma(x))
  $$

Essas derivadas são usadas no cálculo dos gradientes para atualizar os pesos.

---

### **3. Forward Propagation (Propagação Direta)**

Durante o _forward_, a rede calcula:

- **Ativação da camada oculta:**

  $$
  z^{(1)} = X \cdot W^{(1)} + b^{(1)} \quad (\text{soma ponderada})
  $$

  $$
  a^{(1)} = \tanh(z^{(1)}) \quad (\text{aplicação da ativação})
  $$

- **Ativação da camada de saída:**

  $$
  z^{(2)} = a^{(1)} \cdot W^{(2)} + b^{(2)}
  $$

  $$
  a^{(2)} = \sigma(z^{(2)})
  $$

Isso é implementado no método `forward()` da classe `MLP`.

---

### **4. Loss Function (Função de Custo)**

A função de perda usada foi o **erro quadrático médio (MSE)**:

$$
L = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

Na prática, isso aparece em:

```python
loss = np.mean((y - output) ** 2)
```

---

### **5. Backpropagation: Derivadas e Atualização dos Pesos**

A retropropagação consiste em calcular os **gradientes do erro em relação aos pesos**. As fórmulas principais são:

#### 5.1. Erro na camada de saída:

$$
\delta^{(2)} = (a^{(2)} - y) \cdot \sigma'(z^{(2)})
$$

> Implementado como:

```python
erro_saida = (output - y)
delta_saida = erro_saida * sigmoid_derivada(z2)
```

#### 5.2. Gradientes dos pesos da camada de saída:

$$
\frac{\partial L}{\partial W^{(2)}} = (a^{(1)})^T \cdot \delta^{(2)}
$$

> Implementado como:

```python
gradiente_W2 = a1.T @ delta_saida
```

#### 5.3. Erro na camada oculta:

$$
\delta^{(1)} = \delta^{(2)} \cdot (W^{(2)})^T \cdot \tanh'(z^{(1)})
$$

> Implementado como:

```python
delta_oculta = delta_saida @ W2.T * tanh_derivada(z1)
```

#### 5.4. Gradientes dos pesos da camada oculta:

$$
\frac{\partial L}{\partial W^{(1)}} = X^T \cdot \delta^{(1)}
$$

> Implementado como:

```python
gradiente_W1 = X.T @ delta_oculta
```

---

### **6. Atualização dos Pesos (Gradient Descent)**

Com os gradientes computados, aplica-se o passo do gradiente descendente:

$$
W := W - \eta \cdot \frac{\partial L}{\partial W}
\quad
b := b - \eta \cdot \frac{\partial L}{\partial b}
$$

> Implementado como:

```python
self.W2 -= learning_rate * gradiente_W2
self.b2 -= learning_rate * np.sum(delta_saida, axis=0, keepdims=True)
...
```

---

### **7. Conclusão**

As equações fundamentais do **algoritmo de backpropagation** foram fielmente aplicadas no código. A estrutura do programa espelha o pipeline matemático clássico da aprendizagem de redes neurais:

1. **Propagação direta** com soma ponderada e ativação;
2. **Cálculo da perda** via MSE;
3. **Retropropagação do erro** com regras da cadeia;
4. **Atualização dos pesos** com o gradiente descendente.

Este programa serve como uma base sólida para estudos teóricos e práticos de redes neurais.

---

Deseja que eu gere este relatório como um arquivo `.docx` ou `.pdf`? Deseja incluir algum anexo com fórmulas escritas à mão ou exemplos visuais da rede?
