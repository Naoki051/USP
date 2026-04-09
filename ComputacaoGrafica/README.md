# Motor Gráfico 3D - Arquitetura ECS

Este é um motor gráfico 3D didático desenvolvido em **Python** utilizando **Pygame** e **OpenGL**. O projeto utiliza a arquitetura **Entity Component System (ECS)**, garantindo um desacoplamento rigoroso entre dados e lógica.

## 🚀 Funcionalidades

- **Renderização 3D:** Suporte dinâmico a Cubo, Esfera e Tetraedro.
- **Arquitetura Modular (ECS):** Sistemas independentes para Input, Câmera, Luz, Geometria e Material.
- **Iluminação Dinâmica:** Fonte de luz móvel com cálculos de iluminação ambiente, difusa e especular.
- **Câmera Orbital:** Rotação via mouse, Zoom via Scroll e Pan (translação) via setas.
- **Configuração Centralizada:** Ajuste de materiais e constantes de luz via `config.py`.

---

## 🛠️ Requisitos e Instalação

### Python Recomendado
Recomenda-se o uso do **Python 3.10 ou 3.11** para garantir compatibilidade total com as bibliotecas gráficas.

### Dependências
As versões abaixo são necessárias para a estabilidade do contexto OpenGL:
- **pygame**: 2.6.1
- **PyOpenGL**: 3.1.10
- **PyOpenGL-accelerate**: 3.1.10

### Como Instalar (Apenas esta pasta)

Como este projeto faz parte de um repositório acadêmico maior, você pode clonar apenas a pasta `ComputacaoGrafica` usando o comando abaixo:

1. **Inicialize o repositório local:**
   ```bash
   mkdir USP-Grafica && cd USP-Grafica
   git init
   git remote add origin https://github.com/Naoki051/USP.git
   ```

2. **Configure o Sparse Checkout:**
   ```bash
   git config core.sparseCheckout true
   echo "ComputacaoGrafica/*" >> .git/info/sparse-checkout
   ```

3. **Baixe os arquivos da branch correta:**
   ```bash
   git pull origin eps-usp
   cd ComputacaoGrafica
   ```

4. **Ambiente Virtual e Dependências:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS ou venv\Scripts\activate no Windows
   pip install -r requirements.txt
   ```

---

## 🎮 Controles

| Tecla/Mouse | Ação |
| :--- | :--- |
| **Mouse Esq. (Drag)** | Rotacionar a câmera (Órbita) |
| **Mouse Wheel** | Zoom (In/Out) |
| **Setas (L/R/U/D)** | Pan (Mover câmera nos eixos X e Y) |
| **W, A, S, D** | Mover posição da Luz (Eixos X e Y) |
| **Q, E** | Mover posição da Luz (Eixo Z) |
| **Tecla 1** | Instanciar **Cubo** (Cor: Vermelho) |
| **Tecla 2** | Instanciar **Esfera** (Cor: Verde) |
| **Tecla 3** | Instanciar **Tetraedro** (Cor: Azul) |
| **Tecla C** | Aplicar **Cor Aleatória** ao material atual |

---

## 📂 Estrutura do Projeto (ECS)

O projeto segue a separação estrita de responsabilidades:

- **`core/`**: Contém o `EntityManager`, o "cérebro" que gerencia a criação e busca de componentes.
- **`components/`**: Dataclasses que armazenam apenas dados puros (Posição, Cor, Vértices).
- **`systems/`**: Lógica de processamento. Cada sistema atua sobre um grupo específico de componentes (ex: `LightSystem` move apenas entidades com `LightComponent`).
- **`main.py`**: Ponto de entrada que orquestra o loop principal e a comunicação entre os sistemas através de um `InputStateComponent`.

---

## 📝 Notas Técnicas
* **Winding Order:** O tetraedro foi implementado com ordem de vértices anti-horária para garantir o funcionamento correto do *Back-face Culling*.
* **Modularidade:** A adição de novas formas geométricas requer apenas a atualização do `GeometrySystem`, sem necessidade de alterar o motor de renderização.

---

### Autor
Desenvolvido por **Henrique Naoki Teruya** como parte dos estudos de Computação Gráfica na **USP**.