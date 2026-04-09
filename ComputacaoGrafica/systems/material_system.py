import random
from pygame.locals import *
from components.material import MaterialComponent

class MaterialSystem:
    def __init__(self):
        # Mapeamento opcional de cores fixas para facilitar a manutenção
        self.cores_predefinidas = {
            K_1: [1.0, 0.0, 0.0, 1.0],  # Vermelho para o Cubo
            K_2: [0.0, 1.0, 0.0, 1.0],  # Verde para a Esfera
            K_3: [0.0, 0.5, 1.0, 1.0],  # Azul para o Tetraedro
        }

    def update(self, ecs_manager, state):
        """
        Lê o InputStateComponent e aplica mudanças de cor nas entidades 
        que possuem o MaterialComponent.
        """
        
        # 1. Verifica se houve um comando de troca de cor ou objeto
        if state.last_keydown not in [K_1, K_2, K_3, K_c]:
            return

        # 2. Busca todas as entidades que possuem Material
        # No seu caso, o objeto principal, mas poderia afetar múltiplos objetos
        entities = ecs_manager.get_entities_with(MaterialComponent)
        
        for ent_id in entities:
            mat = ecs_manager.get_component(ent_id, MaterialComponent)
            
            # Se for uma tecla de objeto (1, 2, 3), define a cor temática
            if state.last_keydown in self.cores_predefinidas:
                mat.diffuse = self.cores_predefinidas[state.last_keydown]
            
            # Se for a tecla 'C', gera uma cor aleatória
            elif state.last_keydown == K_c:
                mat.diffuse = [
                    random.uniform(0.1, 1.0), 
                    random.uniform(0.1, 1.0), 
                    random.uniform(0.1, 1.0), 
                    1.0
                ]
            
            # Nota: O MaterialComponent tem uma @property 'ambient' que 
            # já se recalcula automaticamente baseada no 'diffuse'.