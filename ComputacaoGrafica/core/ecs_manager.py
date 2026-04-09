# core/ecs_manager.py

class EntityManager:
    def __init__(self):
        # self.entities armazena {eid: {nome_componente: instancia_componente}}
        self.entities = {}
        self.next_id = 0

    def create_entity(self, **components):
        eid = self.next_id
        self.entities[eid] = components
        self.next_id += 1
        return eid

    def get_entities_with(self, *component_types):
        """
        Retorna uma lista de IDs de entidades que possuem 
        TODOS os tipos de componentes solicitados.
        """
        results = []
        for eid, components in self.entities.items():
            # Verifica se cada tipo de componente solicitado está presente nos valores
            has_all = True
            for comp_type in component_types:
                # Checa se alguma instância nos componentes da entidade é do tipo comp_type
                if not any(isinstance(c, comp_type) for c in components.values()):
                    has_all = False
                    break
            
            if has_all:
                results.append(eid)
        return results

    def get_component(self, eid, component_type):
        """
        Retorna a instância do componente de uma entidade específica.
        """
        entity_comps = self.entities.get(eid, {})
        for comp_inst in entity_comps.values():
            if isinstance(comp_inst, component_type):
                return comp_inst
        return None