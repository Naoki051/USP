# config.py
# ==========================================
# CONFIGURAÇÕES GERAIS DA CENA
# ==========================================
# Cor base do objeto principal (RGB)
COR_PADRAO_RGB = (0.0, 0.5, 1.0)

# ==========================================
# CONFIGURAÇÕES DA LUZ (FONTE LUMINOSA)
# ==========================================

# Luz ambiente global da cena.
# Esta luz NÃO tem direção e ilumina todos os objetos igualmente.
# Serve para evitar sombras completamente pretas.
# Valores baixos = cena mais contrastada (sombras mais escuras)
# Valores altos = cena "lavada" (menos profundidade)
LUZ_AMBIENTE = [0.2, 0.2, 0.2, 1.0]


# Luz difusa (iluminação principal).
# Representa a luz que atinge diretamente a superfície e depende do ângulo.
# É a principal responsável pela cor visível do objeto.
# Valores muito altos podem "estourar" a iluminação.
LUZ_DIFUSA = [1.0, 1.0, 1.0, 1.0]


# Luz especular (brilho da fonte de luz).
# Controla a intensidade dos reflexos brilhantes (highlight).
# Branco puro = reflexos fortes e visíveis
# Valores menores = brilho mais suave
LUZ_ESPECULAR = [1.0, 1.0, 1.0, 1.0]

# ==========================================
# CONFIGURAÇÕES DO MATERIAL (OBJETOS 3D)
# ==========================================

# Fator de reflexão da luz ambiente no material.
# Define quanto da luz ambiente o objeto "absorve".
# Exemplo:
# 0.0 = completamente escuro nas sombras
# 0.2 = sombras suaves (recomendado)
# 1.0 = sem contraste (aparência "chapada")
MAT_FATOR_AMBIENTE = 0.5


# Componente especular do material (reflexo do objeto).
# Define o quanto o objeto reflete a luz especular da lâmpada.
#
# Valores baixos (~0.1 - 0.3):
# → aparência fosca (borracha, plástico fosco)
#
# Valores médios (~0.4 - 0.6):
# → plástico comum
#
# Valores altos (~0.8 - 1.0):
# → superfícies polidas ou metálicas
MAT_ESPECULAR = [0.1, 0.1, 0.1, 1.0]


# Brilho do material (shininess).
# Controla o tamanho e a concentração do reflexo especular.
# Intervalo válido: 0 a 128 (OpenGL padrão)
#
# Valores baixos (~5 - 15):
# → reflexo espalhado (fosco)
#
# Valores médios (~20 - 50):
# → plástico ou superfície semi-brilhante
#
# Valores altos (~80 - 128):
# → reflexo pequeno e intenso (metal/polido)
MAT_BRILHO = 10.0