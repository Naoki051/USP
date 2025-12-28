import pandas as pd
import os

def limpar_csv(csv_path):
    try:
        # 1. Garantir que a pasta 'data' existe
        os.makedirs('data', exist_ok=True)

        # 2. Carregar os dados
        df = pd.read_csv(csv_path)
        print(f"Dados carregados de: {csv_path}")

        # 3. Mapeamento das colunas
        mapeamento = {
            'Species': 'species',
            'Culmen Length (mm)': 'bill_length_mm',
            'Culmen Depth (mm)': 'bill_depth_mm',
            'Flipper Length (mm)': 'flipper_length_mm',
            'Body Mass (g)': 'body_mass_g'
        }

        colunas_reais = [c for c in mapeamento.keys() if c in df.columns]
        
        if len(colunas_reais) < 5:
            print(f"Aviso: Colunas encontradas: {df.columns.tolist()}")
            raise ValueError("Não foi possível encontrar as 5 colunas necessárias no CSV.")

        # Selecionar e renomear
        df = df[colunas_reais].rename(columns=mapeamento)

        # 4. Remover linhas com valores ausentes
        df_limpo = df.dropna()
        print(f"Linhas originais: {len(df)} | Após limpeza: {len(df_limpo)}")

        # 5. Salvar na pasta data
        nome_saida = "data/penguins_limpo.csv"
        df_limpo.to_csv(nome_saida, index=False)
        
        print(f"✅ Sucesso! Arquivo salvo em: {nome_saida}")
        return df_limpo

    except FileNotFoundError:
        print(f"❌ Erro: O arquivo '{csv_path}' não foi encontrado.")
    except Exception as e:
        print(f"❌ Erro ao processar os dados: {e}")

# Execução
path = 'data/penguins_lter.csv'
df_final = limpar_csv(path)