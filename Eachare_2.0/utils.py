# Funções que client e server usam 
import os
def incrementa_clock(peer, mensagem_clock = 0):
    clock = peer['clock']
    peer['clock'] = max(clock,mensagem_clock)+1
    print(f'Atualizando clock para {peer["clock"]}')

def adiciona_vizinho(peer, vizinho_info, vizinho_status, vizinho_clock):
    
    vizinhos = peer.get('vizinhos', {})
    caminho_vizinhos = peer.get('vizinhos_path')

    if vizinho_info in vizinhos:
        print(f'Vizinho "{vizinho_info}" já existe. Não adicionado.')
        return False

    peer['vizinhos'][vizinho_info] = {
        'status': vizinho_status,
        'clock': vizinho_clock
    }

    peer['vizinhos'] = vizinhos
    print(f'Adicionando novo vizinho "{vizinho_info}" status "{vizinho_status}" clock "{vizinho_clock}".')

    try:
        # Garante que o diretório exista antes de tentar abrir o arquivo
        os.makedirs(os.path.dirname(caminho_vizinhos), exist_ok=True)
        with open(caminho_vizinhos, 'a') as f:
            f.write(f'{vizinho_info}\n')
    except Exception as e:
        print(f'Erro inesperado ao salvar vizinho "{vizinho_info}" em arquivo: {e}')
        return False

    return True

def atualiza_vizinho(peer, vizinho_info, vizinho_clock, vizinho_status):
    vizinhos = peer.get('vizinhos', {}) 
    if vizinho_info not in vizinhos:
        return adiciona_vizinho(peer, vizinho_info, vizinho_status, vizinho_clock)
    vizinho_existente = vizinhos[vizinho_info]
    if vizinho_existente['clock'] > vizinho_clock:
        return
    if vizinho_existente['status'] == vizinho_status:
        return
    print(f'Atualizando status vizinho "{vizinho_info}" para {vizinho_status}')
    vizinho_existente['clock'] = vizinho_clock
    vizinho_existente['status'] = vizinho_status
