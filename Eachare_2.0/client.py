# Módulo para enviar mensagens para outros peers e interpretar suas respostas
import socket
from utils import incrementa_clock, atualiza_vizinho # Funções utilitárias

# --- Funções de Envio de Mensagens ---
def enviar_mensagem(peer_local: dict, tipo_mensagem: str, args_mensagem: list, peer_destino: str):
    """
    Constrói e envia uma mensagem para um peer de destino, e aguarda a resposta.

    Args:
        peer_local (dict): Dicionário contendo as informações do peer local (peer_info, clock, vizinhos, etc.).
        tipo_mensagem (str): O tipo da mensagem a ser enviada (ex: 'HELLO', 'GET_PEERS', 'DL').
        args_mensagem (list): Lista de argumentos adicionais para a mensagem.
        peer_destino (str): A string de informação do peer de destino (ex: "host:porta").

    Returns:
        bool or dict or None: Depende do tipo de mensagem e resposta.
                              Retorna True para BYE/HELLO.
                              Retorna um dicionário para LS_LIST ou FILE.
                              Retorna None em caso de erro ou resposta desconhecida.
    """
    # Incrementa o clock do peer local antes de enviar a mensagem
    incrementa_clock(peer_local)

    info_peer_local = peer_local['peer_info']
    clock_peer_local = peer_local['clock']
    
    # Constrói a mensagem completa no formato padronizado
    mensagem_completa = f"{info_peer_local} {clock_peer_local} {tipo_mensagem} {' '.join(args_mensagem)}".strip()
    
    print(f"Enviando mensagem: '{mensagem_completa[:100]}' para {peer_destino}")
    
    host_destino, porta_destino = peer_destino.split(":")
    
    try:
        # Cria um socket TCP/IP para o cliente e garante que ele será fechado
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as socket_cliente:
            # Tenta conectar ao peer de destino
            socket_cliente.connect((host_destino, int(porta_destino)))
            
            # Atualiza o clock do vizinho antes de enviar a mensagem
            vizinho_clock_atual = peer_local['vizinhos'].get(peer_destino, {}).get('clock', 0)
            atualiza_vizinho(peer_local, peer_destino, vizinho_clock_atual + 1, 'ONLINE')

            # Envia a mensagem codificada em UTF-8
            socket_cliente.sendall(mensagem_completa.encode('utf-8'))

            # Para mensagens 'BYE' e 'HELLO', não esperamos resposta
            if tipo_mensagem in ['BYE', 'HELLO']:
                
                return True 
            
            # Recebe a resposta do peer de destino
            resposta_bytes = socket_cliente.recv(4096)
            if not resposta_bytes:
                print(f"[CLIENTE] Nenhuma resposta recebida de {peer_destino}.")
                return None

            resposta_str = resposta_bytes.decode('utf-8')
            print(f"Resposta recebida: '{resposta_str[:100]}' de {peer_destino}")

            # Parseia a resposta recebida
            partes_resposta = resposta_str.split(' ')
            if len(partes_resposta) < 3:
                print(f"[CLIENTE] Formato de resposta inválido de {peer_destino}: '{resposta_str}'")
                return None

            origem_info_resp = partes_resposta[0]
            origem_clock_resp = int(partes_resposta[1])
            tipo_resposta = partes_resposta[2]
            args_resposta = partes_resposta[3:] if len(partes_resposta) > 3 else []
            
            # Atualiza o clock local e o status do vizinho com base na resposta
            incrementa_clock(peer_local, origem_clock_resp)
            atualiza_vizinho(peer_local, origem_info_resp, origem_clock_resp, 'ONLINE')

            # Delega a interpretação da resposta para a função `client_handler`
            return client_handler(peer_local, origem_info_resp, origem_clock_resp, tipo_resposta, args_resposta)

    except ConnectionRefusedError:
        print(f"[CLIENTE] Conexão recusada por {peer_destino}. Peer pode estar offline ou inacessível.")
        atualiza_vizinho(peer_local, peer_destino, 0, 'OFFLINE') # Marcar como offline
        return None
    except socket.timeout:
        print(f"[CLIENTE] Tempo limite excedido ao conectar ou receber de {peer_destino}.")
        atualiza_vizinho(peer_local, peer_destino, 0, 'OFFLINE')
        return None
    except Exception as e:
        print(f"[CLIENTE] Erro inesperado ao enviar mensagem para {peer_destino}: {e}")
        return None

## Funções de Manipulação de Respostas do Cliente

def client_handler(peer_local, origem_info_resp, clock_destino, tipo_resposta, args_resposta):
    """
    Lida com a interpretação e processamento de diferentes tipos de respostas recebidas de outros peers.

    Args:
        peer_local (dict): Dicionário contendo as informações do peer local.
        origem_info_resp (str): Informações do peer que enviou a resposta.
        clock_destino (int): Clock do peer que enviou a resposta.
        tipo_resposta (str): O tipo da resposta (ex: 'PEER_LIST', 'LS_LIST', 'FILE', 'ERROR').
        args_resposta (list): Argumentos associados à resposta.

    Returns:
        bool or dict or None: Dependendo do tipo de resposta:
                              - True para 'PEER_LIST' (indica sucesso na atualização).
                              - Dicionário de arquivos para 'LS_LIST'.
                              - Dicionário de chunk de arquivo para 'FILE'.
                              - None em caso de erro ou tipo desconhecido.
    """
    match tipo_resposta:
        case 'PEER_LIST':
            # Processa a lista de vizinhos recebida
            for vizinho_infos_str in args_resposta[1:]: # Ignora o primeiro arg que é a contagem
                try:
                    partes_vizinho = vizinho_infos_str.split(':')
                    if len(partes_vizinho) == 4:
                        info_vizinho = f"{partes_vizinho[0]}:{partes_vizinho[1]}"
                        status_vizinho = partes_vizinho[2]
                        clock_vizinho = int(partes_vizinho[3])
                        # Atualiza as informações do vizinho no estado local do peer
                        atualiza_vizinho(peer_local, info_vizinho, clock_vizinho, status_vizinho)
                    else:
                        print(f"[CLIENTE] Formato de vizinho inválido recebido de {origem_info_resp}: {vizinho_infos_str}")
                except (ValueError, IndexError) as e:
                    print(f"[CLIENTE] Erro ao processar vizinho da lista de {origem_info_resp}: '{vizinho_infos_str}' - {e}")
            return True # Indica que a lista de peers foi processada

        case 'LS_LIST':
            # Processa a lista de arquivos recebida
            arquivos_recebidos = {}
            for arquivo_info_str in args_resposta[1:]: # Ignora o primeiro arg que é a contagem
                try:
                    nome_arquivo, tamanho_arquivo = arquivo_info_str.split(':')
                    # Armazena o arquivo e de qual peer ele veio
                    arquivos_recebidos[(nome_arquivo, int(tamanho_arquivo))] = origem_info_resp
                except ValueError:
                    print(f"[CLIENTE] Formato de arquivo inválido recebido de {origem_info_resp}: '{arquivo_info_str}'")
            return arquivos_recebidos # Retorna o dicionário de arquivos

        case 'FILE':
            # Processa um chunk de arquivo recebido
            if len(args_resposta) == 4:
                arquivo_nome = args_resposta[0]
                arquivo_tamanho_real = int(args_resposta[1]) # Tamanho real do chunk
                arquivo_indice = int(args_resposta[2])
                dados_brutos_b64 = args_resposta[3]
                return {arquivo_indice: dados_brutos_b64} # Retorna o chunk em base64
            else:
                print(f"[CLIENTE] Formato de mensagem FILE inválido recebido de {origem_info_resp}: {args_resposta}")
                return None

        case 'ERROR':
            # Lida com mensagens de erro do servidor
            codigo_erro = args_resposta[0] if args_resposta else "DESCONHECIDO"
            mensagem_erro = ' '.join(args_resposta[1:]) if len(args_resposta) > 1 else "Sem detalhes."
            print(f"[CLIENTE] Erro do servidor {origem_info_resp}: Código {codigo_erro} - {mensagem_erro}")
            return None # Indica que ocorreu um erro

        case _: # Tipo de resposta desconhecido
            print(f"[CLIENTE] Tipo de resposta desconhecido de {origem_info_resp}: '{tipo_resposta}' com argumentos: {args_resposta}")
            return None