import os
import sys
import threading
from cliente import enviar_mensagem
from servidor import iniciar_servidor
from utils import listar_arquivos


def carregar_vizinhos(peer_local):
    """Carrega os vizinhos do peer a partir do arquivo especificado."""
    try:
        with open(peer_local['vizinhos_file'], 'r') as f:
            for linha in f:
                vizinho = linha.strip()
                if vizinho and vizinho != peer_local['peer_info']:
                    peer_local['vizinhos'][vizinho] = {'clock': 0, 'status': 'OFFLINE'}
    except FileNotFoundError:
        print(f"Aviso: Arquivo de vizinhos '{peer_local['vizinhos_file']}' não encontrado.")
    except Exception as e:
        print(f"Erro ao carregar vizinhos: {e}")


def exibir_lista_de_peers(peer_local):
    """Exibe a lista de peers conhecidos com status e clock."""
    peers_ordenados = list(peer_local['vizinhos'].items())
    while True:
        print("\nListar peers:")
        print("         Peer                |   Status   |   Clock")
        print("[0] Voltar ao menu anterior  |            |")
        for i, (vizinho, detalhes) in enumerate(peers_ordenados, start=1):
            print(f"[{i}] \t{vizinho:<20} | {detalhes['status']:<10} | {detalhes['clock']:<4}")
        opcao = input("> ")
        if opcao == '0':
            return
        try:
            int_opcao = int(opcao)
            if 1 <= int_opcao <= len(peers_ordenados):
                peer_destino = peers_ordenados[int_opcao - 1][0]  # Endereço do peer
                enviar_mensagem(peer_local, 'HELLO', [], peer_destino)
                return
            else:
                print("Número fora do intervalo!")
        except ValueError:
            print("Entrada inválida! Digite um número.")


def exibir_arquivos_peers(peer_local, arquivos_peers):
    """Exibe a lista de arquivos encontrados nos peers e permite ao usuário selecionar um para download."""
    while True:
        if arquivos_peers is None:
            print("Nenhum arquivo encontrado com os termos de busca.")
            break
        print("\nArquivos encontrados na rede:")
        print("[0] <Cancelar>                     |            | Peer")
        print("     Nome                         |   Tamanho  | Peer")
        for i, resultado in enumerate(arquivos_peers):
            print(f"[{i+1:2}] {resultado['nome']:<30} | {resultado['tamanho']:<10} | {resultado['peer']}")
        opcao = input("> ")
        try:
            int_opcao = int(opcao)
            if int_opcao == 0:
                break
            elif 1 <= int_opcao <= len(arquivos_peers):
                arquivo_selecionado = arquivos_peers[int_opcao - 1]
                peer_local['tam_arquivo_solicitado'] = int(arquivo_selecionado['tamanho'])
                enviar_mensagem(peer_local, 'DL', [arquivo_selecionado['nome'], '0', '0'], arquivo_selecionado['peer'])
                break # Voltar ao menu principal após solicitar o download
            else:
                print("Valor inválido!")
        except ValueError:
            print("Entrada inválida!")
        
def main():
    if len(sys.argv) != 4:
        print("Uso: python main.py <endereco:porta> <vizinhos.txt> <arquivos>")
        sys.exit(1)

    peer_info = sys.argv[1]
    vizinhos_file = sys.argv[2]
    arquivos_path = sys.argv[3]

    peer_local = {
        'peer_info': peer_info,
        'vizinhos_file': vizinhos_file,
        'vizinhos': {},
        'clock': 0,
        'arquivos_path': arquivos_path
    }

    # Criar a pasta de arquivos se não existir
    os.makedirs(peer_local['arquivos_path'], exist_ok=True)
    print(f"Pasta criada com sucesso: {peer_local['arquivos_path']}")

    carregar_vizinhos(peer_local)

    server_thread = threading.Thread(target=iniciar_servidor, args=(peer_local,), daemon=True)
    server_thread.start()

    while True:
        print("\nEscolha um comando:")
        print("[1] Listar peers conhecidos")
        print("[2] Obter lista de peers de outro peer")
        print("[3] Listar arquivos locais")
        print("[4] Buscar arquivos")
        print("[5] Exibir estatisticas")
        print("[6] Alterar tamanho de chunk")
        print("[9] Sair")
        comando = input("> ").strip()

        if comando == '1':
            exibir_lista_de_peers(peer_local)

        elif comando == '2':
            for peer in list(peer_local['vizinhos']):
                if peer_local['vizinhos'][peer]['status'] == 'ONLINE':
                    enviar_mensagem(peer_local, 'GET_PEERS', [], peer)

        elif comando == '3':
            arquivos = listar_arquivos(peer_local['arquivos_path'])
            if arquivos:
                print("Arquivos locais:")
                for nome, tamanho in arquivos.items():
                    print(f"{nome:<30} | {tamanho} bytes")
            else:
                print("Nenhum arquivo encontrado.")
            input("Pressione Enter para voltar ao menu principal.")

        elif comando == '4':
            arquivos_peers = []
            for peer_info, detalhes in peer_local['vizinhos'].items():
                if detalhes['status'] == 'ONLINE':
                    arquivos_recebidos = enviar_mensagem(peer_local, 'LS', [], peer_info)
                    if arquivos_recebidos and peer_info in arquivos_recebidos:
                        arquivos_do_peer = arquivos_recebidos[peer_info]
                        for nome, tamanho in arquivos_do_peer.items():
                            arquivos_peers.append({'nome': nome, 'tamanho': tamanho, 'peer': peer_info})
                    else:
                        print(f"Aviso: Não foi possível obter a lista de arquivos de {peer_info}.")
            exibir_arquivos_peers(peer_local, arquivos_peers)
        elif comando == '5':
            print("Comando nao implementado.")
        elif comando == '6':
            print("Comando nao implementado.")
        elif comando == '9':
            for vizinho in peer['vizinhos']:
                enviar_mensagem(peer,'BYE','',vizinho)
            print("Encerrando o servidor.")
            break
        else:
            print(f"Comando desconhecido: '{comando}'")


if __name__ == "__main__":
    main()
