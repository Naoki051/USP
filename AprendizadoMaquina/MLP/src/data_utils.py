import pandas as pd
import os

def preparar_dados(caminho):
    if not os.path.exists(caminho):
        raise FileNotFoundError(caminho)

    df = pd.read_csv(caminho)
    df = df.rename(columns={
        'Species': 'species',
        'Culmen Length (mm)': 'bl',
        'Culmen Depth (mm)': 'bd',
        'Flipper Length (mm)': 'fl',
        'Body Mass (g)': 'bm'
    }).dropna()

    X = df[['bl', 'bd', 'fl', 'bm']].values
    y = pd.get_dummies(df['species']).values
    classes = sorted(df['species'].unique())

    x_min, x_max = X.min(axis=0), X.max(axis=0)
    X_norm = (X - x_min) / (x_max - x_min)

    return X_norm, y, x_min, x_max, classes
