from dataclasses import dataclass
from OpenGL.GL import *
from OpenGL.GLU import *

@dataclass
class GeometryComponent:
    primitive: str  # 'cube', 'sphere', 'grid'
    size: float = 1.0
    # Para formas complexas como o Cubo, guardamos os dados aqui
    vertices: list = None
    faces: list = None
    normals: list = None
    edges: list = None