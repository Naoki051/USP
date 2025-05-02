import re

def is_valid_ip_address(address):
    """Verifica se uma string é um endereço IPv4 válido."""
    pattern = r"^(([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])\.){3}([0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$"
    return bool(re.match(pattern, address))

def is_valid_port(port_str):
    """Verifica se uma string representa uma porta válida (inteiro entre 0 e 65535)."""
    try:
        port = int(port_str)
        return 0 <= port <= 65535
    except ValueError:
        return False

def vizinhos_txt_to_dict(vizinhos_file='vizinhos.txt'):
    """
    Lê um arquivo de texto contendo informações de vizinhos e retorna um dicionário
    indexado. A chave do dicionário é um índice incremental, e o valor é um
    dicionário contendo peer (endereço e porta), o status e o clock.
    """
    vizinhos = {}
    indice = 0
    try:
        with open(vizinhos_file, 'r', encoding='utf-8') as arquivo:
            for linha in arquivo:
                peer_vizinho = linha.strip()
                dados = linha.split(":")
                if len(dados) == 2:
                    endereco = dados[0].strip()
                    porta_str = dados[1].strip()
                    if is_valid_ip_address(endereco) and is_valid_port(porta_str):
                        try:
                            vizinhos[peer_vizinho] = {'status': 'OFFLINE', 'clock': 0}
                            indice += 1
                        except Exception as e:
                            print(f"Erro ao adicionar vizinho {peer_vizinho}: {e}")
                    else:
                        print(f"Formato inválido de endereço ou porta na linha: {linha} do arquivo {vizinhos_file}.")
                else:
                    print(f"Linha mal formatada: '{linha}' no arquivo {vizinhos_file}. Formato esperado: 'endereco:porta'.")
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {vizinhos_file}")
    return vizinhos
    
def incrementa_clock(peer,msg_clock = 0):
    """
    Incrementa o clock lógico do peer e imprime o novo valor.
    """
    peer['clock'] = max(msg_clock,peer['clock']) +1
    print(f"=> Atualizando relogio para {peer['clock']}") 
    return

def separar_msg(mensagem):
    """
    Separa uma mensagem recebida em suas partes componentes que serão interpretadas.
    """
    try:
        partes = mensagem.split(" ")
        if len(partes) < 3:
            raise ValueError("Mensagem mal formatada: faltam partes essenciais.")
        origem = partes[0].split(":")
        if len(origem) != 2:
            raise ValueError("Endereço de origem mal formatado.")
        endereco_origem = origem[0]
        porta_origem = int(origem[1])
        clock_origem = int(partes[1])
        mensagem_tipo = partes[2]
        args = list(filter(None, partes[3:])) if len(partes) > 3 else []  # args como lista vazia se ausentes
        mensagem_separada = {
            "endereco_origem": endereco_origem,
            "porta_origem": porta_origem,
            "clock_origem": clock_origem,
            "tipo": mensagem_tipo,
            "args": args
        }
        return mensagem_separada
    except ValueError as e:
        print(f"Erro ao separar mensagem: {e}")
        return None  # Retorna None em caso de erro
    except IndexError:
        print("Erro ao separar mensagem: Mensagem mal formatada.")
        return None
    except Exception as e:
        print(f"Erro inesperado ao separar mensagem: {e}")
        return None

def adiciona_vizinho(peer, endereco, porta, status, clock):
    """
    Adiciona um novo vizinho ao dicionário de vizinhos do peer e ao arquivo de vizinhos.
    Usa o 'peer' (endereco:porta) como chave no dicionário.
    """
    peer_info = f"{endereco}:{porta}"
    try:
        peer['vizinhos'][peer_info] = {"status": status, "clock": clock}
        with open(peer['vizinhos_file'], 'a') as arquivo_vizinhos:
            arquivo_vizinhos.write(f"\n{endereco}:{porta}")
        print(f"Adicionando novo peer {peer_info} {status} {clock}")

    except Exception as e:
        print(f"Erro ao adicionar vizinho {endereco}:{porta}: {e}")

def atualiza_status_vizinho(peer, endereco, porta, status, clock=0):
    """
    Atualiza o status de um vizinho existente no dicionário de vizinhos do peer.
    Se o vizinho não existir, ele é adicionado ao dicionário.
    """
    try:
        peer_info = f"{endereco}:{porta}"
        if peer_info in peer['vizinhos']:
            vizinho = peer['vizinhos'][peer_info]
            # Atualiza clock apenas se o novo for maior
            clock = max(vizinho.get('clock'), int(clock))
            if vizinho.get('status') != status:
                print(f"Atualizando peer {peer_info} status {status}")
            peer['vizinhos'][peer_info] = {
                'status': status,
                'clock': clock
            }
            return True
        # Se não encontrou, adiciona como novo vizinho
        adiciona_vizinho(peer, endereco, porta, status, clock)
        return True
    except Exception as e:
        print(f"Erro inesperado ao atualizar status do vizinho: {e}")
        return False
