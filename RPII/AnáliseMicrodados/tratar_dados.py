MICRODADOS_DE_2024 = r'MICRODADOS_CADASTRO_CURSOS_2024.CSV'

import pandas as pd

df = pd.read_csv(MICRODADOS_DE_2024, sep=';', encoding='latin1', low_memory=False)

df_licenciatura = df[df['TP_GRAU_ACADEMICO'] == 2]
df_licenciatura_letras = df_licenciatura[df_licenciatura['NO_CURSO'].str.contains('letra', case=False)]

colunas_to_drop =['NU_ANO_CENSO', 'NO_REGIAO', 'NO_UF', 'SG_UF', 'NO_MUNICIPIO', 
                  'NO_CURSO', 'NO_CINE_ROTULO', 'NO_CINE_AREA_GERAL', 'NO_CINE_AREA_ESPECIFICA', 
                  'NO_CINE_AREA_DETALHADA']

df_licenciatura_letras = df_licenciatura_letras.drop(columns=colunas_to_drop)
df_licenciatura_letras = df_licenciatura_letras.dropna()

df_licenciatura_letras.to_csv('df_licenciatura_letras.csv', index=False)


