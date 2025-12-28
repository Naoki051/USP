import matplotlib.pyplot as plt
import numpy as np

def plotar_kfold_completo(scores, historicos):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    for i, h in enumerate(historicos):
        ax1.plot(h['val'], label=f'Fold {i+1}')
    ax1.set_title('Loss de Validação')
    ax1.legend()
    ax1.grid(True)

    ax2.bar(range(len(scores)), scores)
    ax2.axhline(np.mean(scores), linestyle='--', color='red')
    ax2.set_title('Acurácia por Fold')

    plt.tight_layout()
    plt.show()
