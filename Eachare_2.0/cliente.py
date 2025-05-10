import socket
import os
import base64
from utils import *

def enviar_mensagem(peer, tipo, args, peer_destino):
    mensagem = construir_mensagem(peer, tipo, args)
    print(f"Enviando mensagem: '{mensagem}' para {peer_destino}")
    incrementa_clock(peer)
    host, port = peer_destino.split(":")

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
            client_socket.connect((host, int(port)))
            atualiza_status_peer(peer, peer_destino, 'ONLINE', peer['vizinhos'][peer_destino]['clock']+1)
            client_socket.sendall(mensagem.encode())

            resposta = client_socket.recv(1024)
            if resposta:
                return client_handler(peer, resposta.decode('utf-8'), client_socket)

    except Exception as e:
        print(f"[CLIENT] Erro ao enviar para {peer_destino}: {e}")
        atualiza_status_peer(peer, peer_destino, 'OFFLINE', peer['vizinhos'][peer_destino]['clock']+1)

    return False  # Falha ou sem resposta

def client_handler(peer, resposta, client_socket):
    print(f"Resposta recebida: '{resposta[:67]}'...")
    origem_info, origem_clock, tipo, args = separar_mensagem(resposta)

    if not origem_info:
        print(f"[CLIENT] Erro: Mensagem mal formatada: '{resposta}'")
        return False

    incrementa_clock(peer,origem_info, origem_clock)
    atualiza_status_peer(peer, origem_info, 'ONLINE',origem_clock)

    match tipo:
        case 'PEER_LIST':
            return processa_peer_list(peer, args)
        case 'LS_LIST':
            return processa_ls_list(origem_info, args)
        case 'FILE':
            return processa_file(peer, args, client_socket)
        case _:
            print(f"[CLIENT] Erro: Tipo de mensagem não reconhecido: '{tipo}'")
            return False

def processa_peer_list(peer, args):
    if not args:
        print("[CLIENT] Aviso: Lista de peers vazia.")
        return False

    try:
        num_peers = int(args[0])
        dados_peers = args[1:]

        if len(dados_peers) != num_peers:
            print("[CLIENT] Erro: Quantidade de peers não bate com o informado.")
            return False

        for dados_peer in dados_peers:
            try:
                ip, port, status, clock = dados_peer.split(':')
                peer_info = f"{ip}:{port}"
                clock = int(clock)
                atualiza_status_peer(peer, peer_info, status,clock)
                peer['vizinhos'][peer_info]['clock'] = clock
            except Exception as e:
                print(f"[CLIENT] Erro ao processar dados do peer '{dados_peer}': {e}")
                return False

        return True

    except ValueError:
        print(f"[CLIENT] Erro: Valor inválido no cabeçalho da PEER_LIST ({args[0]}).")
        return False

def processa_ls_list(origem_info, args):
    if not args:
        print(f"[CLIENT] Aviso: Lista de arquivos vazia de {origem_info}.")
        return False

    try:
        num_arquivos = int(args[0])
        dados_arquivos = args[1:]

        if len(dados_arquivos) != num_arquivos:
            print(f"[CLIENT] Erro: Quantidade de arquivos não bate com o informado para {origem_info}.")
            return False

        arquivos = {origem_info: {}}
        for dado in dados_arquivos:
            try:
                nome, tamanho = dado.split(':')
                arquivos[origem_info][nome] = tamanho
            except ValueError:
                print(f"[CLIENT] Aviso: Formato incorreto para arquivo: '{dado}' de {origem_info}")

        return arquivos

    except ValueError:
        print(f"[CLIENT] Erro: Cabeçalho LS_LIST inválido de {origem_info}: '{args[0]}'")
        return False
    
def processa_file(peer, args, client_socket):
    if len(args) != 4:
        print("[CLIENT] Erro: Esperados 4 argumentos na mensagem FILE.")
        return False

    nome_arquivo = args[0]
    conteudo_base64 = args[3]
    conteudo_completo = base64.b64decode(conteudo_base64)
    bytes_recebidos = len(conteudo_completo)

    try:
        while bytes_recebidos < peer['tam_arquivo_solicitado']:
            dados_recebidos = client_socket.recv(1024)
            novos_dados = base64.b64decode(dados_recebidos)
            conteudo_completo += novos_dados
            bytes_recebidos += len(novos_dados)

        caminho_destino = os.path.join(peer["arquivos_path"], nome_arquivo)
        with open(caminho_destino, "wb") as f:
            f.write(conteudo_completo)
        print(f"Download do arquivo {nome_arquivo} finalizado.")
        if bytes_recebidos > peer['tam_arquivo_solicitado']:
            print('algo deu errado!')
        return True

    except base64.binascii.Error:
        print("[CLIENT] Erro: Falha na decodificação base64.")
    except Exception as e:
        print(f"[CLIENT] Erro ao salvar arquivo '{nome_arquivo}': {e}")

    return False
