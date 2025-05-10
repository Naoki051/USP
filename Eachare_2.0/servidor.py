import socket, os, base64
import threading
from utils import *

def iniciar_servidor(peer):
    """
    Esta função inicializa o servidor de um peer específico.

    """
    host, port = peer['peer_info'].split(":")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, int(port)))
    server_socket.listen()

    try:
        running = True
        while running:
            conexao, endereco = server_socket.accept()
            thread = threading.Thread(target=conn_handler, args=(peer, conexao, endereco))
            thread.daemon = True
            thread.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Encerrando servidor.")
    finally:
        server_socket.close()

def conn_handler(peer, conexao, endereco):
    """
    Esta função é executada em uma thread separada para lidar com cada conexão de cliente.

    """
    try:
        dados = conexao.recv(1024)
        if not dados:
            return

        mensagem = dados.decode('utf-8')
        print(f"Mensagem recebida: '{mensagem}'")

        origem_info, origem_clock, tipo, args = separar_mensagem(mensagem)

        if not origem_info:
            print(f"[SERVER] Erro: Mensagem mal formatada de {endereco}")
            return

        incrementa_clock(peer, origem_info, origem_clock)
        atualiza_status_peer(peer, origem_info, 'ONLINE',origem_clock)

        resposta = gerar_resposta(peer, origem_info, tipo, args, conexao)
        if resposta:
            conexao.sendall(resposta.encode())
            print(f"Enviando resposta: '{resposta[:67]}...' para {origem_info}")
        else:
            print(f"[SERVER] Nenhuma resposta enviada para {origem_info}.")

    except Exception as e:
        print(f"[SERVER] Erro ao tratar conexão de {endereco}: {e}")
    finally:
        conexao.close()


def gerar_resposta(peer, origem_info, tipo, args, conexao):
    """
    Esta função gera uma resposta com base no tipo de mensagem recebida.

    """
    match tipo:
        case 'GET_PEERS':
            vizinhos = [
                f"{id}:{info['status']}:{info['clock']}"
                for id, info in peer['vizinhos'].items()
                if id != origem_info
            ]
            return construir_mensagem(peer, 'PEER_LIST', [str(len(vizinhos))] + vizinhos)

        case 'LS':
            arquivos = listar_arquivos(peer['arquivos_path'])
            args = [str(len(arquivos))] + [f"{nome}:{tamanho}" for nome, tamanho in arquivos.items()]
            return construir_mensagem(peer, 'LS_LIST', args)

        case 'DL':
            if len(args) != 3:
                print(f"[SERVER] Erro: DL inválido.")
                return

            nome_arquivo = args[0]
            caminho = os.path.join(peer['arquivos_path'], nome_arquivo)
            if not os.path.isfile(caminho):
                print(f"[SERVER] Arquivo solicitado não encontrado: '{nome_arquivo}'")
                return
            try:
                with open(caminho, 'rb') as f:
                    dados_b64 = base64.b64encode(f.read()).decode('utf-8')
                    return construir_mensagem(peer, 'FILE', [nome_arquivo, '0', '0', dados_b64])

            except Exception as e:
                print(f"[SERVER] Erro ao enviar arquivo '{nome_arquivo}': {e}")

    return None