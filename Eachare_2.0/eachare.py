# Exibir menus e realizar ações de usuários

from client import enviar_mensagem
from server import iniciar_servidor
import numpy as np
import sys, os, threading, base64, time
import concurrent.futures

def carregar_vizinhos(peer_info, vizinhos_path):
    dict_vizinhos = {}
    try:
        with open(vizinhos_path, 'r') as f:
            for linha in f:
                vizinho_info = linha.strip()
                if vizinho_info and vizinho_info != peer_info:
                    dict_vizinhos[vizinho_info] = {'clock': 0, 'status': 'OFFLINE'}
        return dict_vizinhos
    except Exception as e:
        print(f"Erro ao carregar vizinhos: {e}")
        return dict_vizinhos

def carregar_arquivos(arquivos_path):
    arquivos = {}
    try:
        for nome_arquivo in os.listdir(arquivos_path):
            caminho_arquivo = os.path.join(arquivos_path, nome_arquivo)
            if os.path.isfile(caminho_arquivo):
                tamanho = os.path.getsize(caminho_arquivo)
                arquivos[nome_arquivo] = tamanho
        return arquivos
    except Exception as e:
        print(f"Erro ao listar arquivos em '{arquivos_path}': {e}")
        return arquivos

def exibir_menu():
    print("\nEscolha um comando:")
    print("[1] Listar peers conhecidos")
    print("[2] Obter lista de peers de outro peer")
    print("[3] Listar arquivos locais")
    print("[4] Buscar arquivos")
    print("[5] Exibir estatisticas")
    print("[6] Alterar tamanho de chunk")
    print("[9] Sair")
    escolha = input('> ')
    return escolha

def exibir_vizinhos(peer):
    vizinhos = peer['vizinhos']
    lista_de_vizinhos = list(vizinhos.keys())

    while True:
        print("\t    Peer               | Status     | Clock")
        print("\t-----------------------|------------|-------")
        print("\t[0] Voltar             | ---        | ---")

        for index, vizinho in enumerate(lista_de_vizinhos,1):
            print(f"\t[{index}] {vizinho:<18} | {vizinhos[vizinho]['status']:<11}| {vizinhos[vizinho]['clock']}")

        opcao = input("> ")
        if opcao == '0':
            return
        try:
            int_opcao = int(opcao)
            peer_destino = lista_de_vizinhos[int_opcao - 1]
            enviar_mensagem(peer,'HELLO',[],peer_destino)
            return
        except ValueError:
            print('Entrada inválida!')
        except IndexError:
            print('Escolha um valor listado!')
        except Exception as e:
            print(f"Erro ao listar peers: {e}")

def get_peers(peer):
    vizinhos = peer['vizinhos']
    lita_vizinhos = list(vizinhos)
    for vizinho in lita_vizinhos:
        enviar_mensagem(peer,'GET_PEERS',[],vizinho)

def exibir_arquivos_locais(peer):
    peer['arquivos'] = carregar_arquivos(peer['arquivos_path'])
    arquivos = peer['arquivos']
    print('\t   Nome                        | Tamnaho')
    print('\t-------------------------------|--------')
    for arquivo in arquivos:
        print(f"\t - {arquivo:<27} | {arquivos[arquivo]}")

def buscar_arquivos_na_rede(peer: dict):
    """
    Busca arquivos disponíveis em todos os vizinhos online do peer local.

    Args:
        peer (dict): Dicionário contendo as informações do peer local.
                     Espera-se que 'peer' tenha a chave 'vizinhos'.
    """
    arquivos_da_rede = {} # Estrutura: { (nome_arquivo, tamanho): [peer_origem_1, peer_origem_2, ...] }

    for vizinho_info, vizinho_data in peer.get('vizinhos', {}).items():
        if vizinho_data.get('status') == 'ONLINE':
            print(f"[REDE] Solicitando lista de arquivos de {vizinho_info}...")
            # Envia uma mensagem 'LS' (List Files) para o vizinho
            # 'enviar_mensagem' deve retornar um dicionário no formato { (nome, tamanho): peer_origem }
            arquivos_recebidos = enviar_mensagem(peer, 'LS', [], vizinho_info)

            if isinstance(arquivos_recebidos, dict): # Verifica se a resposta é um dicionário de arquivos
                for (nome_arquivo, tamanho_arquivo), origem_peer in arquivos_recebidos.items():
                    chave_arquivo = (nome_arquivo, tamanho_arquivo)
                    if chave_arquivo not in arquivos_da_rede:
                        arquivos_da_rede[chave_arquivo] = []
                    # Adiciona o peer que possui o arquivo à lista de origens
                    if origem_peer not in arquivos_da_rede[chave_arquivo]:
                        arquivos_da_rede[chave_arquivo].append(origem_peer)

    if not arquivos_da_rede:
        print("Nenhum arquivo encontrado na rede no momento.")
        return # Sai da função se não houver arquivos

    exibir_arquivos_da_rede(peer, arquivos_da_rede)

def exibir_arquivos_da_rede(peer: dict, arquivos_da_rede: dict):
    """
    Exibe os arquivos encontrados na rede e permite ao usuário selecionar um para download.

    Args:
        peer (dict): Dicionário contendo as informações do peer local.
        arquivos_da_rede (dict): Dicionário de arquivos disponíveis na rede.
                                 Formato: { (nome_arquivo, tamanho): [peer_origem_1, ...] }
    """
    lista_de_arquivos = list(arquivos_da_rede.keys()) # Converte as chaves do dicionário em uma lista para indexação

    while True:
        print("\n" + "="*40)
        print("Arquivos encontrados na rede:")
        print("    No. | Nome              | Tamanho (bytes) | Peers Disponíveis")
        print("    ----|-------------------|-----------------|------------------")
        print("    [0] | Cancelar Download |                 |")
        
        for index, (nome, tam) in enumerate(lista_de_arquivos, 1):
            peers_disponiveis = ", ".join(arquivos_da_rede[(nome, tam)])
            print(f"    [{index}] | {nome:<17} | {tam:<15} | {peers_disponiveis}")
        
        print("="*40)
        opcao = input('Digite o número do arquivo para fazer o download (0 para cancelar): ').strip()

        if opcao == '0':
            print("Download cancelado pelo usuário.")
            return

        try:
            int_opcao = int(opcao)
            if 1 <= int_opcao <= len(lista_de_arquivos):
                arquivo_escolhido = lista_de_arquivos[int_opcao - 1]
                nome_arquivo, tam_arquivo = arquivo_escolhido
                lista_de_peers = arquivos_da_rede[arquivo_escolhido]
                
                requisitar_download(peer, nome_arquivo, tam_arquivo, lista_de_peers)
                return # Sai da função após iniciar o download
            else:
                print("⚠️ Opção inválida. Digite um número da lista.")
        except ValueError:
            print("⚠️ Entrada inválida. Por favor, digite um número.")
        except IndexError:
            print("⚠️ Número fora do intervalo. Por favor, digite um número válido.")
        except Exception as e:
            print(f"⚠️ Ocorreu um erro inesperado: {e}")


def requisitar_download(peer: dict, nome_arquivo: str, tam_arquivo: int, lista_de_peers: list):
    if not lista_de_peers:
        print(f"[DOWNLOAD] Erro: Não há peers disponíveis para baixar '{nome_arquivo}'.")
        return

    tam_chunk = peer['chunk']
    num_chunks = int(np.ceil(tam_arquivo / tam_chunk))
    dados_completos_dict = {}  # Usar um dicionário para armazenar chunks por índice

    start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:  # Você pode ajustar max_workers
        future_to_chunk_index = {
            executor.submit(
                enviar_mensagem,
                peer,
                'DL',
                [nome_arquivo, str(tam_chunk), str(indice)],
                lista_de_peers[indice % len(lista_de_peers)]
            ): indice
            for indice in range(num_chunks)
        }

        for future in concurrent.futures.as_completed(future_to_chunk_index):
            indice = future_to_chunk_index[future]
            try:
                dados_recebidos = future.result()
                if indice in dados_recebidos:
                    dados_completos_dict[indice] = base64.b64decode(dados_recebidos[indice])
                    print(f"[DOWNLOAD] Chunk {indice+1}/{num_chunks} baixado com sucesso.")
                else:
                    print(f"[DOWNLOAD] Erro: Chunk {indice+1}/{num_chunks} não encontrado nos dados recebidos.")
            except Exception as e:
                print(f"[DOWNLOAD] Erro ao baixar chunk {indice+1}/{num_chunks}: {e}")
    end = time.time()
    tempo = end-start
    # Reconstruir os dados completos na ordem correta
    dados_completos = b''
    if not (tam_chunk,len(lista_de_peers),tam_arquivo) in peer['downloads']:
        peer['downloads'][(tam_chunk,len(lista_de_peers),tam_arquivo)] = [tempo]
    else:
        peer['downloads'][(tam_chunk,len(lista_de_peers),tam_arquivo)].append(tempo)
    for i in range(num_chunks):
        if i in dados_completos_dict:
            dados_completos += dados_completos_dict[i]
        else:
            print(f"[DOWNLOAD] Aviso: Chunk {i+1}/{num_chunks} está faltando. O arquivo pode estar incompleto.")
            return False # Indicar falha se houver chunks faltando

    salvar_arquivo(peer, nome_arquivo, dados_completos)
    return True


def salvar_arquivo(peer: dict, nome: str, dados_completos: bytes):
    """
    Salva os dados completos de um arquivo no diretório de downloads do peer.

    Args:
        peer (dict): Dicionário contendo as informações do peer local.
                     Espera-se a chave 'arquivos_path' para o diretório de destino.
        nome (str): Nome do arquivo a ser salvo.
        dados_completos (bytes): Conteúdo binário completo do arquivo.
    """
    # Define o caminho de destino para o arquivo
    # Usa 'downloads' como diretório padrão caso 'arquivos_path' não esteja configurado
    caminho_destino_base = peer.get("arquivos_path", "downloads")
    caminho_completo_arquivo = os.path.join(caminho_destino_base, nome)

    # Garante que o diretório de destino exista
    os.makedirs(os.path.dirname(caminho_completo_arquivo), exist_ok=True)

    try:
        with open(caminho_completo_arquivo, "wb") as f:
            f.write(dados_completos)
            
        print(f"✅ Download do arquivo '{nome}' finalizado e salvo em '{caminho_completo_arquivo}'.")
    except OSError as e:
        print(f"❌ Erro de sistema de arquivos ao salvar '{nome}': {e}")
    except Exception as e:
        print(f"❌ Ocorreu um erro inesperado durante o salvamento do arquivo '{nome}': {e}")


def exibir_estatisticas(peer):
    downloads = peer['downloads']
    
    print(f"Estatísticas:") # Adiciona um ID do peer se disponível
    print(f" Chunk  | Peers | Tam. Arquivo | Amostras | Tempo Médio(s) | Desvio Padrão")
    print( ' -------|-------|--------------|----------|----------------|--------------') 
    for chave, tempos in downloads.items():
        chunk = chave[0]
        n_peers = chave[1]
        tam_arquivo = chave[2]
        if not tempos:  # Verifica se a lista de tempos não está vazia para evitar erro de divisão por zero
            media_tempos = 0
            desvio = 0
        else:
            media_tempos = sum(tempos) / len(tempos)
            desvio = np.std(tempos)
        print(f" {chunk:<6} | {n_peers:<5} | {tam_arquivo:<12} | {len(tempos):<8} | {media_tempos:<14.4f} | {desvio:.4f}")

def alterar_chunk(peer):
    print('Digite novo tamanho de chunk:')
    chunk = input('> ')
    try:
        int_chunk = int(chunk)
        if int_chunk == 0:
            print("Tamanho do chunk não alterado!")
            return
        if int_chunk > 4096:
            print("Tamanho do chunk não pode ser maior que 4096!")
            return
        if not np.log2(int_chunk).is_integer():
            print("Tamanho do chunk deve ser uma potência de 2!")
            return

        print(f'Tamanho de chunk alterado: {int_chunk}')
        peer['chunk'] = int_chunk
    except:
        print('Valor inálido! Voltando ao menu...')

def main():
    if len(sys.argv) != 4:
        print("Uso: python eachare.py <endereco:porta> <vizinhos.txt> <arquivos>")
        sys.exit(1)

    peer_info = sys.argv[1]
    vizinhos_path = sys.argv[2]
    arquivos_path = sys.argv[3]
    peer_vizinhos = carregar_vizinhos(peer_info, vizinhos_path)

    peer_arquivos = carregar_arquivos(arquivos_path)
    peer_clock = 1
    chunk = 256
    
    downloads = {}

    peer = {
        'peer_info': peer_info,
        'arquivos_path': arquivos_path,
        'vizinhos_path': vizinhos_path,
        'vizinhos' : peer_vizinhos,
        'arquivos': peer_arquivos,
        'clock': peer_clock,
        'chunk': chunk,
        'downloads': downloads
    }
    
    thread_servidor = threading.Thread(target=iniciar_servidor, args=(peer,), daemon=True)
    thread_servidor.start()

    comandos = {
        '1': lambda: exibir_vizinhos(peer),
        '2': lambda: get_peers(peer),
        '3': lambda: exibir_arquivos_locais(peer),
        '4': lambda: buscar_arquivos_na_rede(peer),
        '5': lambda: exibir_estatisticas(peer),
        '6': lambda: alterar_chunk(peer),
    }

    while True:
        comando = exibir_menu()
        if comando == '9':
            print("Encerrando o servidor.")
            break
        elif comando in comandos:
            comandos[comando]()
        else:
            print(f"Comando desconhecido: '{comando}'")

if __name__ == "__main__":
    main()