# components/camera.py
from dataclasses import dataclass

@dataclass
class CameraComponent:
    rot_x: float = 0.0
    rot_y: float = 0.0
    zoom: float = -5.0
    pan_x: float = 0.0
    pan_y: float = 0.0
    sensibilidade: float = 0.5
    is_dragging: bool = False

# components/light.py
@dataclass
class LightSourceComponent:
    pos_x: float = 2.0
    pos_y: float = 2.0
    pos_z: float = 2.0
    color: list = (1.0, 1.0, 1.0, 1.0)