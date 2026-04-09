# Motor Gráfico 3D — ECS (Python + OpenGL)

Um motor gráfico 3D educacional desenvolvido em **Python**, utilizando **Pygame + PyOpenGL**, com arquitetura baseada em **Entity Component System (ECS)**.

O objetivo do projeto é demonstrar, de forma prática, conceitos de **computação gráfica moderna**, **arquitetura de engines** e **desacoplamento de sistemas**.

---

## 🛠️ Funcionalidades

### Renderização 3D
* Cubo, Esfera e Tetraedro.
* Geometrias procedurais e configuráveis.
* Pipeline de renderização baseado em OpenGL.

### Arquitetura ECS
* Entidades totalmente desacopladas.
* Componentes de dados puros (Dataclasses).
* Sistemas independentes de processamento.

### Iluminação Dinâmica
* Cálculos de luz ambiente, difusa e especular (Phong Shading).
* Fonte de luz móvel em tempo real.

### Câmera Orbital
* Rotação com mouse (drag).
* Zoom com scroll.
* Pan com teclado (setas).

---

## ⚙️ Instalação e Execução

### Requisitos
* Python **3.10 ou 3.11** (recomendado para compatibilidade com OpenGL-accelerate).
* Bibliotecas: `pygame`, `PyOpenGL`, `PyOpenGL-accelerate`, `numpy`.

### 1. Clonando e Acessando o Projeto
```bash
git clone https://github.com/Naoki051/USP.git
cd USP/ComputacaoGrafica
```

### 2. Ambiente Virtual e Dependências

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Executando o Motor
```bash
python main.py
```

---

## 🎮 Controles

| Entrada      | Ação                      |
| ------------ | ------------------------- |
| Mouse (drag) | Rotação da câmera         |
| Scroll       | Zoom                      |
| Setas        | Pan da câmera             |
| **W A S D** | Movimentação da luz (X/Y) |
| **Q E** | Movimentação da luz (Z)   |
| **1** | Instanciar Cubo           |
| **2** | Instanciar Esfera         |
| **3** | Instanciar Tetraedro      |
| **C** | Cor aleatória             |

---

## 🏗️ Arquitetura do Projeto (ECS)

### Estrutura de Pastas
```
core/        → Engine ECS (EntityManager e gerenciamento base)
components/  → Dados puros (Transform, Light, Material, etc.)
systems/     → Lógica de processamento (Render, Input, Camera, etc.)
main.py      → Loop principal da aplicação e orquestração
```

### Filosofia do Sistema
* **Components:** Não possuem lógica, apenas armazenamento de estado.
* **Systems:** Não armazenam estado persistente, apenas processam entidades que possuem os componentes necessários.
* **Entities:** São apenas IDs numéricos que agrupam componentes.

---

## 📝 Notas Técnicas

### Geometria e Renderização
* **Winding Order:** Tetraedro implementado com ordem anti-horária para suporte correto a *Back-face Culling*.
* **Topologia:** Renderizador dinâmico com suporte automático a faces triangulares e quadriláteros.

### Extensibilidade
Para adicionar novas geometrias, o design permite:
1. Definir os novos dados (vértices/faces) no `GeometrySystem`.
2. Registrar a nova primitiva.
3. **Nenhuma alteração** é necessária no `RenderSystem`, mantendo o código de desenho protegido.

---

## 👨‍💻 Autor

Desenvolvido por **Henrique Naoki Teruya**.
Projeto acadêmico de **Computação Gráfica — USP**.
