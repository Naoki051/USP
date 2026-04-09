# systems/input_system.py
import pygame
from components.controls import InputStateComponent

class InputSystem:
    def update(self, ecs_manager):
        input_ent = ecs_manager.get_entities_with(InputStateComponent)[0]
        state = ecs_manager.get_component(input_ent, InputStateComponent)
        
        state.last_keydown = None
        state.wheel = 0
        state.mouse_rel = (0, 0)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: state.quit_requested = True
            if event.type == pygame.MOUSEWHEEL: state.wheel = event.y
            if event.type == pygame.MOUSEMOTION: state.mouse_rel = event.rel
            if event.type == pygame.KEYDOWN: state.last_keydown = event.key
        
        state.keys = pygame.key.get_pressed()
        state.mouse_buttons = pygame.mouse.get_pressed()