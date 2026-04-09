# Motor Gráfico 3D - Arquitetura ECS

Este é um motor gráfico 3D didático desenvolvido em **Python** utilizando **Pygame** e **OpenGL**. O projeto foi migrado de uma estrutura Orientada a Objetos para **Entity Component System (ECS)**, permitindo maior desacoplamento entre dados (Componentes) e lógica (Sistemas).

## 🚀 Funcionalidades

- **Renderização 3D:** Suporte a Cubo, Esfera e Tetraedro.
- **Arquitetura Modular (ECS):** Separação total entre Input, Câmera, Luz, Geometria e Material.
- **Iluminação Dinâmica:** Fonte de luz móvel com cálculos de ambiente, difusa e especular.
- **Câmera Orbital:** Rotação via mouse, Zoom via Scroll e Pan via setas do teclado.
- **Configuração Centralizada:** Ajuste de materiais e luzes via `config.py`.

---

## 🛠️ Requisitos e Instalação

### Python Recomendado
Para evitar incompatibilidades com o `PyOpenGL-accelerate`, recomenda-se o uso do **Python 3.10 ou 3.11**.

### Dependências
As versões abaixo são críticas para a estabilidade do contexto OpenGL no sistema:

- **pygame**: 2.6.1
- **PyOpenGL**: 3.1.10
- **PyOpenGL-accelerate**: 3.1.10 (Opcional, mas recomendado para performance)

### Como instalar

1. Clone o repositório:
   ```bash
   git clone https://github.com/seu-usuario/seu-repositorio.git
   cd seu-repositorio
   ```

2. Crie um ambiente virtual (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/macOS
   # ou
   venv\Scripts\activate     # Windows
   ```

3. Instale as dependências:
   ```bash
   pip install pygame==2.6.1 PyOpenGL==3.1.10 PyOpenGL-accelerate==3.1.10
   ```

---

## 🎮 Controles

| Tecla/Mouse | Ação |
| :--- | :--- |
| **Mouse Esq. (Drag)** | Rotacionar a câmera ao redor do objeto |
| **Mouse Wheel** | Zoom (Aproximar/Afastar) |
| **Setas (L/R/U/D)** | Pan (Mover a câmera lateralmente/verticalmente) |
| **W, A, S, D** | Mover a posição da Luz (Eixos X e Y) |
| **Q, E** | Mover a posição da Luz (Eixo Z) |
| **Tecla 1** | Trocar para o **Cubo** (e cor padrão) |
| **Tecla 2** | Trocar para a **Esfera** (e cor padrão) |
| **Tecla 3** | Trocar para o **Tetraedro** (e cor padrão) |
| **Tecla C** | Alterar para uma **Cor Aleatória** |

---

## 📂 Estrutura do Projeto (ECS)

```text
├── main.py              # Orquestrador do loop e inicialização das entidades
├── config.py            # Constantes de cores, luzes e materiais
├── components/          # Dados Puros (Dataclasses)
│   ├── camera.py
│   ├── geometry.py
│   ├── light.py
│   ├── material.py
│   └── transform.py
├── systems/             # Lógica e Processamento
│   ├── camera_system.py
│   ├── geometry_system.py
│   ├── input_system.py
│   ├── light_system.py
│   ├── material_system.py
│   └── render_system.py # O único sistema com chamadas OpenGL de desenho
└── core/                # Motor base
    └── ecs_manager.py   # Gerenciador de Entidades e busca de componentes
```

---

## 📝 Notas de Versão
* O uso de `PyOpenGL-accelerate` pode exigir compiladores C++ instalados no sistema (como o Build Tools do Visual Studio no Windows). Caso tenha erro na instalação dele, o motor funcionará apenas com o `PyOpenGL` comum, porém com menos performance em malhas complexas.

---

### Autor
Desenvolvido como projeto de estudo de Computação Gráfica e Arquitetura de Software.