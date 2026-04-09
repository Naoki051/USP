# systems/render_system.py
from OpenGL.GL import *
from OpenGL.GLU import *

class RenderSystem:
    def __init__(self, largura, altura):
        # Configuração do Quadric (usado para esferas e cilindros)
        self.quadric = gluNewQuadric()
        gluQuadricNormals(self.quadric, GLU_SMOOTH)
        
        # Inicialização da Matrix de Projeção
        glViewport(0, 0, largura, altura)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, (largura / altura), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        
        glEnable(GL_DEPTH_TEST)

    def update(self, ecs_manager, camera_data):
        # 1. Limpeza de Buffer
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # 2. Configuração da Câmera (View Transformation)
        glTranslatef(camera_data.pan_x, camera_data.pan_y, camera_data.zoom)
        glRotatef(camera_data.rot_x, 1, 0, 0)
        glRotatef(camera_data.rot_y, 0, 1, 0)

        # 3. Processamento de Luzes (Light Systems)
        # Primeiro configuramos todas as luzes antes de desenhar os objetos
        self._setup_lights(ecs_manager)

        # 4. Renderização de Objetos (Geometry Systems)
        entities = ecs_manager.entities
        for ent_id, comps in entities.items():
            trans = comps.get('transform')
            mat = comps.get('material')
            geom = comps.get('geometry')

            # Pula entidades que não são objetos 3D renderizáveis
            if not all([trans, mat, geom]): 
                continue

            glPushMatrix()
            glTranslatef(trans.x, trans.y, trans.z)

            # --- Passo A: Renderização Sólida (Faces) ---
            glEnable(GL_LIGHTING)
            glMaterialfv(GL_FRONT, GL_AMBIENT, mat.ambient)
            glMaterialfv(GL_FRONT, GL_DIFFUSE, mat.diffuse)
            glMaterialfv(GL_FRONT, GL_SPECULAR, mat.specular)
            glMaterialfv(GL_FRONT, GL_SHININESS, mat.shininess)

            self._draw_geometry(geom)
            
            # --- Passo B: Renderização de Arestas (Wireframe/Edges) ---
            glDisable(GL_LIGHTING)
            glColor3fv(mat.edge_color)
            self._draw_edges(geom)

            glPopMatrix()

    def _setup_lights(self, ecs_manager):
        """Configura o estado das luzes e desenha as esferas indicadoras"""
        # Desabilita iluminação global antes de reconfigurar as luzes ativas
        glDisable(GL_LIGHTING) 
        
        for ent_id, comps in ecs_manager.entities.items():
            light = comps.get('light')
            trans = comps.get('transform')

            if not light or not trans:
                continue

            # Configura a luz no OpenGL
            glEnable(GL_LIGHTING)
            glEnable(light.light_id)
            
            pos = [trans.x, trans.y, trans.z, 1.0] # 1.0 = Positional light
            glLightfv(light.light_id, GL_POSITION, pos)
            glLightfv(light.light_id, GL_AMBIENT, light.ambient)
            glLightfv(light.light_id, GL_DIFFUSE, light.diffuse)
            glLightfv(light.light_id, GL_SPECULAR, light.specular)
            
            # Atenuação
            glLightf(light.light_id, GL_CONSTANT_ATTENUATION, light.constant_attenuation)
            glLightf(light.light_id, GL_LINEAR_ATTENUATION, light.linear_attenuation)
            glLightf(light.light_id, GL_QUADRATIC_ATTENUATION, light.quadratic_attenuation)

            # Desenha a esfera visual da lâmpada
            if light.show_visual_sphere:
                glPushMatrix()
                glDisable(GL_LIGHTING)
                glTranslatef(trans.x, trans.y, trans.z)
                glColor3fv(light.color_visual)
                gluQuadricDrawStyle(self.quadric, GLU_FILL)
                gluSphere(self.quadric, light.sphere_radius, 16, 16)
                glEnable(GL_LIGHTING)
                glPopMatrix()

    def _draw_geometry(self, geom):
        if geom.primitive == 'sphere':
            gluQuadricDrawStyle(self.quadric, GLU_FILL)
            gluSphere(self.quadric, geom.size, 32, 32)
        
        elif geom.primitive == 'cube':
            # VERIFICAÇÃO DINÂMICA: 
            # Se a primeira face tem 3 vértices, usa TRIANGLES. Se tem 4, usa QUADS.
            if not geom.faces: return
            
            num_vertices = len(geom.faces[0])
            mode = GL_TRIANGLES if num_vertices == 3 else GL_QUADS
            
            glBegin(mode)
            for i, face in enumerate(geom.faces):
                # Aplica a normal da face se existir
                if i < len(geom.normals):
                    glNormal3fv(geom.normals[i])
                
                for v_idx in face:
                    glVertex3fv(geom.vertices[v_idx])
            glEnd()

    def _draw_edges(self, geom):
        glLineWidth(1.0)
        if geom.primitive == 'sphere':
            gluQuadricDrawStyle(self.quadric, GLU_LINE)
            # Offset leve para evitar Z-fighting com a face sólida
            gluSphere(self.quadric, geom.size * 1.001, 16, 16)
        
        elif geom.primitive == 'cube':
            glBegin(GL_LINES)
            for edge in geom.edges:
                for v_idx in edge: 
                    glVertex3fv(geom.vertices[v_idx])
            glEnd()
            
        elif geom.primitive == 'grid':
            t = int(geom.size)
            glBegin(GL_LINES)
            for i in range(-t, t + 1):
                glVertex3f(i, 0.001, -t); glVertex3f(i, 0.001, t)
                glVertex3f(-t, 0.001, i); glVertex3f(t, 0.001, i)
            glEnd()