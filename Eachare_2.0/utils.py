import os

def construir_mensagem(peer, tipo, args):
    """
    Constrói uma mensagem formatada para envio a outros peers.

    A mensagem segue o formato:
    '[peer_info] [clock] [tipo] [arg1] [arg2] ...'

    Args:
        peer (dict): Contém 'peer_info' (str) e 'clock' (int).
        tipo (str): Tipo da mensagem (ex: 'HELLO', 'PEER_LIST').
        args (list): Lista de strings como argumentos da mensagem.

    Returns:
        str: Mensagem formatada.
    """
    return f'{peer["peer_info"]} {peer["clock"]} {tipo} {" ".join(args)}'.strip()

def separar_mensagem(mensagem):
    """
    Separa uma mensagem recebida em origem, clock, tipo e argumentos.

    Esperado: '[origem] [clock] [tipo] [args...]'

    Args:
        mensagem (str): Mensagem recebida.

    Returns:
        tuple:
            - origem_info (str|None)
            - origem_clock (int|None)
            - tipo (str|None)
            - args (list[str])
    """
    partes = mensagem.strip().split()
    if len(partes) < 3:
        print(f"[UTIL] Erro: Mensagem mal formatada: '{mensagem}'")
        return None, None, None, []

    origem_info = partes[0]
    try:
        origem_clock = int(partes[1])
    except ValueError:
        print(f"[UTIL] Erro: Clock inválido na mensagem: '{partes[1]}'")
        return None, None, None, []

    tipo = partes[2]
    args = partes[3:]
    return origem_info, origem_clock, tipo, args

def atualiza_status_peer(peer, vizinho_peer, vizinho_status, vizinho_clock):
    """
    Atualiza o status de um vizinho na lista de vizinhos.

    Args:
        peer (dict): Contém 'vizinhos'.
        vizinho_peer (str): Endereço do vizinho (host:porta).
        status (str): Novo status ('ONLINE', 'OFFLINE').
    """
    if vizinho_peer not in peer['vizinhos']:
        adicionar_peer(peer, vizinho_peer, vizinho_status, vizinho_clock)
        return 
    if vizinho_clock < peer['vizinhos'][vizinho_peer]['clock']:
        return 
    if peer['vizinhos'][vizinho_peer]['status'] == vizinho_status:
        return 

    peer['vizinhos'][vizinho_peer]['status'] = vizinho_status
    print(f'Atualizando peer {vizinho_peer} status {vizinho_status}')
    return 

def adicionar_peer(peer, vizinho_peer, vizinho_status='OFFLINE', vizinho_clock = 0):
    """
    Adiciona um novo vizinho à estrutura local e ao arquivo de vizinhos.

    Args:
        peer (dict): Contém 'vizinhos' e 'vizinhos_file'.
        vizinho_peer (str): Endereço do vizinho.
        status (str): Status inicial (padrão: 'OFFLINE').
    """
    print(f"Adicionando peer: {vizinho_peer} {vizinho_status} {vizinho_clock}")
    peer['vizinhos'][vizinho_peer] = {'clock': vizinho_clock, 'status': vizinho_status}
    try:
        with open(peer['vizinhos_file'], 'a') as arquivo:
            arquivo.write(vizinho_peer + '\n')
    except Exception as e:
        print(f"[UTIL] Erro ao salvar vizinho '{vizinho_peer}' no arquivo: {e}")

def incrementa_clock(peer, origem_info='', origem_clock=0):
    """
    Incrementa o clock lógico do peer local.

    Sincroniza com o clock de origem, se fornecido.

    Args:
        peer (dict): Deve conter 'clock' e 'vizinhos'.
        origem_info (str): Endereço do peer de origem.
        origem_clock (int): Clock do peer de origem.
    """
    if origem_clock:
        peer['clock'] = max(peer['clock'], origem_clock)
        if origem_info in peer['vizinhos']:
            peer['vizinhos'][origem_info]['clock'] = max(peer['vizinhos'][origem_info]['clock'], origem_clock)
    peer['clock'] += 1
    print(f"=> Atualizando relogio para {peer['clock']}")

def listar_arquivos(caminho_pasta):
    """
    Lista os arquivos de um diretório com seus tamanhos em bytes.

    Args:
        caminho_pasta (str): Caminho da pasta.

    Returns:
        dict: {nome_arquivo: tamanho_em_bytes}
    """
    arquivos = {}
    try:
        for nome in os.listdir(caminho_pasta):
            caminho = os.path.join(caminho_pasta, nome)
            if os.path.isfile(caminho):
                arquivos[nome] = os.path.getsize(caminho)
    except Exception as e:
        print(f"[UTIL] Erro ao listar arquivos em '{caminho_pasta}': {e}")
    return arquivos
