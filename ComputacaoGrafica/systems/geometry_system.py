from pygame.locals import *
from components.geometry import GeometryComponent

# Dados do Tetraedro (Pirâmide de base triangular)
TETRA_VERTICES = [[0, 1, 0],[-1, -1, 1],[1, -1, 1],[0, -1, -1]]
TETRA_FACES = [(0, 1, 2),(0, 2, 3),(0, 3, 1),(1, 3, 2)]
TETRA_NORMALS = [(0, 0.5, 1),(1, 0.5, -0.5),(-1, 0.5, -0.5),(0, -1, 0)]
TETRA_EDGES = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (3, 1)]

# Dados do Cubo
CUBO_VERTICES = [[1, -1, -1], [1, 1, -1], [-1, 1, -1], [-1, -1, -1], [1, -1, 1], [1, 1, 1], [-1, -1, 1], [-1, 1, 1]]
CUBO_FACES = [(0, 1, 2, 3), (4, 5, 7, 6), (0, 1, 5, 4), (3, 2, 7, 6), (1, 2, 7, 5), (0, 3, 6, 4)]
CUBO_NORMALS = [(0, 0, -1), (0, 0, 1), (1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0)]
CUBO_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 7), (7, 6), (6, 4), (0, 4), (1, 5), (2, 7), (3, 6)]

class GeometrySystem:
    def update(self, ecs_manager, state):
        # Só processa se uma das teclas de troca foi pressionada
        if state.last_keydown not in [K_1, K_2, K_3]: 
            return
        
        # Busca a entidade que possui geometria (nosso objeto principal)
        entities = ecs_manager.get_entities_with(GeometryComponent)
        if not entities: 
            return
            
        geom = ecs_manager.get_component(entities[0], GeometryComponent)

        if state.last_keydown == K_1:
            geom.primitive = 'cube'
            geom.vertices = CUBO_VERTICES
            geom.faces = CUBO_FACES
            geom.normals = CUBO_NORMALS
            geom.edges = CUBO_EDGES
            
        elif state.last_keydown == K_2:
            geom.primitive = 'sphere'
            geom.size = 1.2  # Define o raio da esfera
            
        elif state.last_keydown == K_3:
            # Reutilizamos o modo 'cube' do renderer para desenhar malhas customizadas
            geom.primitive = 'cube' 
            geom.vertices = TETRA_VERTICES
            geom.faces = TETRA_FACES
            geom.normals = TETRA_NORMALS
            geom.edges = TETRA_EDGES