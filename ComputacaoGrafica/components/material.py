from dataclasses import dataclass, field
from config import MAT_FATOR_AMBIENTE, MAT_ESPECULAR, MAT_BRILHO

@dataclass
class MaterialComponent:
    diffuse: list  # [r, g, b, a]
    edge_color: list = field(default_factory=lambda: [0.1, 0.1, 0.1])
    specular: list = field(default_factory=lambda: MAT_ESPECULAR)
    
    # Se MAT_BRILHO for uma lista [10.0], use default_factory
    # Se for apenas 10.0, o código antigo funcionaria.
    shininess: any = field(default_factory=lambda: MAT_BRILHO)

    @property
    def ambient(self):
        return [c * MAT_FATOR_AMBIENTE for c in self.diffuse[:3]] + [1.0]