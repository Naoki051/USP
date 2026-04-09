# components/controls.py
from dataclasses import dataclass

@dataclass
class InputStateComponent:
    """Armazena o estado bruto do hardware para outros sistemas lerem."""
    keys: list = None
    mouse_rel: tuple = (0, 0)
    mouse_buttons: list = None
    wheel: int = 0
    quit_requested: bool = False
    last_keydown: int = None # Armazena a última tecla pressionada neste frame