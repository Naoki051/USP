# systems/camera_system.py
from pygame.locals import *
from components.camera import CameraComponent

class CameraSystem:
    def update(self, ecs_manager, state, sensibilidade=0.5):
        # 1. Recupera a entidade da câmera
        entities = ecs_manager.get_entities_with(CameraComponent)
        if not entities:
            return
            
        cam = ecs_manager.get_component(entities[0], CameraComponent)

        # 2. Rotação (Mouse Drag)
        # O mouse_rel vem do InputStateComponent preenchido pelo InputSystem
        if state.mouse_buttons[0]: # Botão esquerdo pressionado
            cam.rot_x += state.mouse_rel[1] * sensibilidade
            cam.rot_y += state.mouse_rel[0] * sensibilidade
        
        # 3. Zoom (Mouse Wheel)
        if state.wheel != 0:
            cam.zoom += state.wheel * 0.5
            # Clamp para evitar que a câmera inverta ou vá longe demais
            cam.zoom = max(min(cam.zoom, -2.0), -30.0)
        
        # 4. Pan (Setas do Teclado)
        # Movimento Horizontal
        if state.keys[K_LEFT]:  cam.pan_x += 0.1
        if state.keys[K_RIGHT]: cam.pan_x -= 0.1
        
        # Movimento Vertical
        if state.keys[K_UP]:    cam.pan_y -= 0.1
        if state.keys[K_DOWN]:  cam.pan_y += 0.1