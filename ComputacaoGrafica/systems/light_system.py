from pygame.locals import *
from components.light import LightComponent
from components.transform import TransformComponent

class LightSystem:
    def __init__(self, velocidade=0.1):
        self.velocidade = velocidade

    def update(self, ecs_manager, state):
        # Filtra entidades que são luzes E possuem posição (Transform)
        light_ents = ecs_manager.get_entities_with(LightComponent, TransformComponent)
        
        for ent in light_ents:
            trans = ecs_manager.get_component(ent, TransformComponent)
            
            # Movimentação Horizontal (A/D)
            if state.keys[K_a]: trans.x -= self.velocidade
            if state.keys[K_d]: trans.x += self.velocidade
            
            # Movimentação Vertical (W/S)
            if state.keys[K_w]: trans.y += self.velocidade
            if state.keys[K_s]: trans.y -= self.velocidade
            
            # Movimentação de Profundidade (Q/E)
            if state.keys[K_q]: trans.z -= self.velocidade
            if state.keys[K_e]: trans.z += self.velocidade