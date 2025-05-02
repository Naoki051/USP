import socket,os,base64
from modules.utils import separar_msg, incrementa_clock, atualiza_status_vizinho

def start_server(peer):
    """
    Inicia o servidor P2P para receber e processar conexões de outros peers.
    """
    endereco = peer['endereco']
    porta = peer['porta']
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            server_socket.bind((endereco, porta))
            running = True
            while running:
                server_socket.listen()
                conn, addr = server_socket.accept()
                with conn:
                    mensagem = conn.recv(1024).decode()
                    if mensagem:
                        running = conn_handler(conn,peer,mensagem)
    except Exception as e:
        print(f"Erro: {e}")
    
def conn_handler(conn,peer,mensagem):
    """
    Processa mensagens recebidas de outros peers e executa ações apropriadas.
    """
    if mensagem == 'SAIR':
        return False
    print(f"Mensagem recebida: {mensagem}")
    mensagem_partes = separar_msg(mensagem)
    endereco_origem = mensagem_partes['endereco_origem']
    porta_origem = mensagem_partes['porta_origem']
    clock_origem = mensagem_partes['clock_origem']
    tipo = mensagem_partes['tipo']
    incrementa_clock(peer,clock_origem)
    if tipo == 'HELLO':
        atualiza_status_vizinho(peer,endereco_origem,porta_origem,'ONLINE',clock_origem)
    elif tipo == 'GET_PEERS':
        atualiza_status_vizinho(peer,endereco_origem,porta_origem,'ONLINE',clock_origem)
        incrementa_clock(peer,clock_origem)
        resposta = construir_resposta_peers(peer, mensagem_partes)
        print(f'Encaminhando resposta: "{resposta}" para {endereco_origem}:{porta_origem}')
        conn.sendall(resposta.encode())
    elif tipo == 'LS':
        atualiza_status_vizinho(peer,endereco_origem,porta_origem,'ONLINE',clock_origem)
        incrementa_clock(peer,clock_origem)
        resposta = construir_resposta_ls(peer)
        print(f'Encaminhando resposta: "{resposta}" para {endereco_origem}:{porta_origem}')
        conn.sendall(resposta.encode())
    elif tipo == 'DL':
        nome_arquivo = mensagem_partes['args'][0]
        atualiza_status_vizinho(peer,endereco_origem,porta_origem,'ONLINE',clock_origem)
        incrementa_clock(peer,clock_origem)
        resposta = construir_resposta_dl(peer,nome_arquivo)
        print(f'Encaminhando resposta: "{resposta}" para {endereco_origem}:{porta_origem}')
        conn.sendall(resposta.encode())
    elif tipo == 'BYE':
        atualiza_status_vizinho(peer,endereco_origem,porta_origem,'OFFLINE',clock_origem)
    return True

def construir_resposta_peers(peer, mensagem_separada):
    """
    Constrói uma resposta contendo a lista de vizinhos do peer, com exceção do peer de origem.
    """
    vizinhos_formatados = []
    origem_peer_info = f"{mensagem_separada['endereco_origem']}:{mensagem_separada['porta_origem']}"
    for vizinho_peer_info in peer['vizinhos']:
        if vizinho_peer_info != origem_peer_info:
            status = peer['vizinhos'][vizinho_peer_info].get('status', 'OFFLINE')  # Valor padrão se status não estiver presente
            clock = peer['vizinhos'][vizinho_peer_info].get('clock', 0)          # Valor padrão se clock não estiver presente
            vizinhos_formatados.append(f"{vizinho_peer_info}:{status}:{clock}")

    resposta = f"{peer['endereco']}:{peer['porta']} {peer['clock']} PEER_LIST {len(vizinhos_formatados)} {' '.join(vizinhos_formatados)}"
    return resposta

def construir_resposta_ls(peer):
    """
    Constrói uma resposta contendo a lista de arquivos do peer fornecendo o nome e tamanho do arquivo.
    """
    arquivos_info = []
    resposta = ''
    try:
        for nome_arquivo in os.listdir(peer['diretorio_compartilhado']):
            caminho_arquivo = os.path.join(peer['diretorio_compartilhado'], nome_arquivo)
            if os.path.isfile(caminho_arquivo):
                tamanho_arquivo = os.path.getsize(caminho_arquivo)
                arquivos_info.append(f"{nome_arquivo}:{tamanho_arquivo}")
    except FileNotFoundError:
        print(f"Erro: Diretório compartilhado '{peer['diretorio_compartilhado']}' não encontrado.")
    except Exception as e:
        print(f"Erro ao listar arquivos: {e}")

    resposta = f"{peer['endereco']}:{peer['porta']} {peer['clock']} LS_LIST {len(arquivos_info)} {' '.join(arquivos_info)}"
    return resposta

def construir_resposta_dl(peer, nome_arquivo):
    """
    Constrói uma mensagem de resposta DL contendo o conteúdo do arquivo solicitado.

    Args:
        peer (dict): Dicionário contendo informações do peer (endereco, porta, clock).
        nome_arquivo (str): O nome do arquivo a ser enviado (localizado na pasta 'arquivos').

    Returns:
        str: Uma string formatada contendo as informações do peer, tipo de mensagem 'FILE',
             nome do arquivo e o conteúdo do arquivo codificado em Base64.
             Retorna None se o arquivo não for encontrado ou ocorrer um erro na leitura.
    """
    caminho_arquivo = os.path.join('arquivos', nome_arquivo)
    try:
        with open(os.path.join('arquivos', nome_arquivo), 'rb') as f:
            conteudo = f.read()
            conteudo_base64 = base64.b64encode(conteudo).decode('utf-8')
            mensagem = f"{peer['endereco']}:{peer['porta']} {peer['clock']} FILE {nome_arquivo} 0 0 {conteudo_base64}"
            return mensagem
    except FileNotFoundError:
        print(f"Erro: Arquivo '{nome_arquivo}' não encontrado na pasta 'arquivos'.")
        return None
    except Exception as e:
        print(f"Erro ao ler o arquivo '{nome_arquivo}': {e}")
        return None