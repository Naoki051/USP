import pygame
from pygame.locals import *

# ECS Core
from core.ecs_manager import EntityManager

# Sistemas (Agora especializados)
from systems.render_system import RenderSystem
from systems.input_system import InputSystem
from systems.camera_system import CameraSystem
from systems.light_system import LightSystem
from systems.geometry_system import GeometrySystem
from systems.material_system import MaterialSystem

# Componentes
from components.transform import TransformComponent
from components.material import MaterialComponent
from components.geometry import GeometryComponent
from components.camera import CameraComponent
from components.light import LightComponent
from components.controls import InputStateComponent # Novo componente

# Configurações
import config

def main():
    # 1. Inicialização do Pygame e Contexto OpenGL
    pygame.init()
    largura, altura = 1024, 800
    pygame.display.set_mode((largura, altura), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Motor 3D Modular ECS")
    clock = pygame.time.Clock()

    # 2. Instanciação do ECS e Sistemas
    ecs = EntityManager()
    
    # Sistemas de lógica
    input_system = InputSystem()
    camera_system = CameraSystem()
    light_system = LightSystem()
    geometry_system = GeometrySystem()
    material_system = MaterialSystem()
    
    # Sistema de Saída (Render)
    render_system = RenderSystem(largura, altura)

    # 3. Criação das Entidades Base
    
    # Entidade Global de Input (Singleton de estado)
    ent_input = ecs.create_entity(
        state=InputStateComponent()
    )

    # Entidade de CÂMERA
    ent_camera = ecs.create_entity(
        camera=CameraComponent(zoom=-10.0)
    )

    # Entidade de LUZ
    ecs.create_entity(
        transform=TransformComponent(x=2.0, y=3.0, z=2.0),
        light=LightComponent()
    )

    # 4. Criação do OBJETO PRINCIPAL (Cubo inicial)
    cor_difusa = [1.0, 0.0, 0.0, 1.0]
    ecs.create_entity(
        transform=TransformComponent(x=0, y=0, z=0),
        material=MaterialComponent(diffuse=cor_difusa),
        geometry=GeometryComponent(
            primitive='cube',
            vertices=[[1,-1,-1], [1,1,-1], [-1,1,-1], [-1,-1,-1], [1,-1,1], [1,1,1], [-1,-1,1], [-1,1,1]],
            faces=[(0,1,2,3), (4,5,7,6), (0,1,5,4), (3,2,7,6), (1,2,7,5), (0,3,6,4)],
            normals=[(0,0,-1), (0,0,1), (1,0,0), (-1,0,0), (0,1,0), (0,-1,0)],
            edges=[(0,1), (1,2), (2,3), (3,0), (4,5), (5,7), (7,6), (6,4), (0,4), (1,5), (2,7), (3,6)]
        )
    )

    # 5. Loop Principal
    executando = True
    while executando:
        # --- A. CAPTURA (Hardware -> ECS) ---
        input_system.update(ecs)
        
        # Obtemos o estado de input para os sistemas de lógica
        state = ecs.get_component(ent_input, InputStateComponent)
        
        if state.quit_requested:
            executando = False
            break

        # --- B. LÓGICA (ECS -> ECS) ---
        # Cada sistema processa apenas sua responsabilidade baseada no estado do input
        camera_system.update(ecs, state)
        light_system.update(ecs, state)
        geometry_system.update(ecs, state)
        material_system.update(ecs, state)

        # --- C. RENDERIZAÇÃO (ECS -> Tela) ---
        camera_data = ecs.get_component(ent_camera, CameraComponent)
        render_system.update(ecs, camera_data)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()