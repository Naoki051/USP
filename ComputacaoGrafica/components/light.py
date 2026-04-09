from dataclasses import dataclass, field
from OpenGL.GL import GL_LIGHT0
from config import LUZ_AMBIENTE, LUZ_DIFUSA, LUZ_ESPECULAR

@dataclass
class LightComponent:
    # Dados de configuração da luz (OpenGL)
    light_id: int = GL_LIGHT0
    ambient: list = field(default_factory=lambda: list(LUZ_AMBIENTE))
    diffuse: list = field(default_factory=lambda: list(LUZ_DIFUSA))
    specular: list = field(default_factory=lambda: list(LUZ_ESPECULAR))
    
    # Atenuação
    constant_attenuation: float = 1.0
    linear_attenuation: float = 0.0
    quadratic_attenuation: float = 0.05
    
    # Propriedades visuais (a "bolinha" que representa a luz)
    show_visual_sphere: bool = True
    sphere_radius: float = 0.1

    @property
    def color_visual(self):
        """Retorna o RGB da luz difusa para pintar a esfera visual"""
        return self.diffuse[:3]