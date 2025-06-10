# Módulo para receber mensagens de clientes e gerar respostas
import socket
import os
import base64
import threading
from utils import * # Funções utilitárias como incrementa_clock e atualiza_vizinho

# --- Funções do Servidor ---
def iniciar_servidor(peer):
    """
    Inicializa o servidor de um peer específico.
    O servidor escuta por conexões de entrada e as delega para threads.
    """
    host, port = peer['peer_info'].split(":")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, int(port)))
    server_socket.listen()

    print(f"[SERVER] Servidor iniciado em {host}:{port}")

    try:
        running = True
        while running:
            conexao, endereco = server_socket.accept()
            # Inicia uma nova thread para lidar com a conexão
            thread = threading.Thread(target=conn_handler, args=(peer, conexao, endereco))
            thread.daemon = True  # Garante que a thread será encerrada quando o programa principal morrer
            thread.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Encerrando servidor.")
    finally:
        server_socket.close()
        print("[SERVER] Socket do servidor fechado.")

def conn_handler(peer, conexao, endereco):
    """
    Lida com uma conexão de cliente individual em uma thread separada.
    Processa a mensagem recebida, atualiza o clock e gera uma resposta.
    """
    try:
        dados = conexao.recv(4096)
        if not dados:
            return

        mensagem = dados.decode('utf-8')
        print(f"Mensagem recebida: '{mensagem}' de {endereco}")

        # Parseia a mensagem recebida
        partes_mensagem = mensagem.split(' ')
        if len(partes_mensagem) < 3:
            print(f"[SERVER] Mensagem inválida recebida de {endereco}: '{mensagem}'")
            return

        origem_info = partes_mensagem[0]
        origem_clock = int(partes_mensagem[1])
        tipo = partes_mensagem[2]
        args = partes_mensagem[3:] if len(partes_mensagem) > 3 else []

        # Atualiza o clock do peer com base na mensagem recebida
        incrementa_clock(peer, origem_clock)

        # Lógica de tratamento de tipos de mensagem BYE e HELLO
        if tipo == 'BYE':
            atualiza_vizinho(peer, origem_info, origem_clock, 'OFFLINE')
            return 
        
        atualiza_vizinho(peer, origem_info, origem_clock, 'ONLINE')
        if tipo == 'HELLO':
            return

        # Gera a resposta para outros tipos de mensagem
        resposta = gerar_resposta(peer, origem_info, tipo, args)

        if resposta:
            conexao.sendall(resposta.encode('utf-8'))
            print(f"Enviando resposta: '{resposta}' para {origem_info}")

    except Exception as e:
        print(f"[SERVER] Erro ao tratar conexão de {endereco}: {e}")
    finally:
        conexao.close()
        # print(f"[SERVER] Conexão com {endereco} encerrada.") # Pode ser útil para debug

# --- Funções de Mensagens e Respostas ---
def construir_mensagem(peer, tipo, args):
    """
    Constrói uma mensagem padronizada a ser enviada para outros peers.
    Formato: <peer_info> <clock> <tipo> <args...>
    """
    return f"{peer['peer_info']} {peer['clock']} {tipo} {' '.join(args)}".strip()

def gerar_resposta(peer, origem_info, tipo, args):
    """
    Gera uma resposta com base no tipo de mensagem recebida.
    Utiliza um 'match-case' para lidar com diferentes tipos de requisições.
    """
    match tipo:
        case 'GET_PEERS':
            # Retorna uma lista de vizinhos (exceto o próprio remetente)
            vizinhos = [
                f"{id}:{info['status']}:{info['clock']}"
                for id, info in peer['vizinhos'].items()
                if id != origem_info
            ]
            return construir_mensagem(peer, 'PEER_LIST', [str(len(vizinhos))] + vizinhos)

        case 'LS':
            # Retorna uma lista de arquivos locais com seus tamanhos
            arquivos = listar_arquivos(peer) # Passa o dicionário 'peer' para acessar 'arquivos_path'
            # Converte a lista de tuplas (nome, tamanho) para o formato de string desejado
            args_resposta = [str(len(arquivos))] + [f"{nome}:{tamanho}" for nome, tamanho in arquivos]
            return construir_mensagem(peer, 'LS_LIST', args_resposta)

        case 'DL':
            # Lida com a requisição de download de um chunk de arquivo
            if len(args) != 3:
                print(f"[SERVER] Erro: Requisição DL inválida. Formato esperado: 'DL <nome_arquivo> <tamanho_chunk> <index_chunk>'")
                return None

            nome_arquivo = args[0]
            try:
                tamanho_chunk = int(args[1])
                index_chunk = int(args[2])
            except ValueError:
                print(f"[SERVER] Erro: Tamanho ou índice do chunk inválido na requisição DL.")
                return None
            
            offset = index_chunk * tamanho_chunk
            caminho_arquivo = os.path.join(peer['arquivos_path'], nome_arquivo)

            if not os.path.isfile(caminho_arquivo):
                print(f"[SERVER] Arquivo solicitado não encontrado: '{nome_arquivo}' em {caminho_arquivo}")
                return None
            
            try:
                with open(caminho_arquivo, 'rb') as f:
                    f.seek(offset) # Move o ponteiro do arquivo para o início do chunk
                    dados_brutos_chunk = f.read(tamanho_chunk) # Lê apenas o chunk solicitado
                    
                    tamanho_real_lido = len(dados_brutos_chunk) # O tamanho real pode ser menor no último chunk
                    dados_chunk_b64 = base64.b64encode(dados_brutos_chunk).decode('utf-8')
                    
                    # Formato da resposta FILE: nome_arquivo tamanho_real_lido indice_chunk dados_base64
                    args_resposta = [nome_arquivo, str(tamanho_real_lido), str(index_chunk), dados_chunk_b64]
                    return construir_mensagem(peer,'FILE',args_resposta)

            except Exception as e:
                print(f"[SERVER] Erro ao enviar arquivo '{nome_arquivo}': {e}")
                return None
        case _:
            # Caso de tipo de mensagem não reconhecido
            print(f"[SERVER] Tipo de mensagem desconhecido: '{tipo}'")
            return None

# --- Funções de Manipulação de Arquivos Locais ---
def listar_arquivos(peer):
    """
    Lista os arquivos presentes no diretório 'arquivos_path' do peer.
    Retorna uma lista de tuplas (nome_arquivo, tamanho_em_bytes).
    """
    caminho_arquivos = peer['arquivos_path']
    arquivos_info = []
    try:
        for nome_arquivo in os.listdir(caminho_arquivos):
            caminho_completo_arquivo = os.path.join(caminho_arquivos, nome_arquivo)
            # Verifica se é um arquivo (e não um subdiretório)
            if os.path.isfile(caminho_completo_arquivo):
                tamanho = os.path.getsize(caminho_completo_arquivo)
                arquivos_info.append((nome_arquivo, tamanho))
        return arquivos_info
    except FileNotFoundError:
        print(f"[SERVER] Erro: O diretório de arquivos '{caminho_arquivos}' não foi encontrado.")
        return []
    except Exception as e:
        print(f"[SERVER] Erro ao listar arquivos em '{caminho_arquivos}': {e}")
        return []