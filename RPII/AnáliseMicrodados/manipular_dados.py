map_organizacao_academica = {
    1: 'Universidade',
    2: 'Centro Universitário',
    3: 'Faculdade',
    4: 'Instituto Federal de Educação, Ciência e Tecnologia',
    5: 'Centro Federal de Educação Tecnológica'
}

map_rede_ensino = {
    1: 'Pública',
    2: 'Privada'
}

map_categoria_administrativa = {
    1: 'Pública Federal',
    2: 'Pública Estadual',
    3: 'Pública Municipal',
    4: 'Privada com fins lucrativos',
    5: 'Privada sem fins lucrativos',
    6: 'Privada - Particular em sentido estrito', # Usada somente em 2009
    7: 'Especial', # Criada em 2012
    8: 'Privada comunitária', # Usada somente em 2009
    9: 'Privada confessional' # Usada somente em 2009
}

map_modalidade_ensino = {
    1: 'Presencial',
    2: 'Curso a distância (EaD)'
}

map_tipo_dimensao = {
    1: 'Cursos presenciais ofertados no Brasil',
    2: 'Cursos a distância ofertados no Brasil',
    3: 'Cursos a distância com dimensão de dados somente a nível Brasil',
    4: 'Cursos a distância ofertados por instituições brasileiras no exterior'
}

map_in_capital = {
    1: 'Sim',
    0: 'Não'  # Pressupondo que o valor 0 significa 'Não'
}

map_in_gratuito = {
    1: 'Sim',
    0: 'Não' # Pressupondo que o valor 0 significa 'Não'
}

map_nivel_academico = {
    1: 'Graduação',
    2: 'Sequencial de Formação Específica'
}

colunas_to_ignore = [
    'CO_IES', 'CO_CURSO','QT_SIT_TRANCADA', 'QT_SIT_DESVINCULADO', 
    'QT_SIT_TRANSFERIDO', 'QT_SIT_FALECIDO']


