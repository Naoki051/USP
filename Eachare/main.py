import time
import sys
import os
from peer import Peer

# =========================
# UTILIDADES DE UI
# =========================

def limpar_tela():
    os.system("cls" if os.name == "nt" else "clear")

def pause():
    input("\n⏎ Pressione ENTER para continuar...")

def header(peer):
    print("=" * 50)
    print(" 🌐 P2P FILE SHARING")
    print("-" * 50)
    print(f" Peer: {peer.host}:{peer.port}")
    print(f" Vizinhos: {len(peer.vizinhos)} | Chunk: {peer.chunk_size} bytes")
    print("=" * 50)

def titulo(texto):
    print("\n" + "-" * 50)
    print(f" {texto}")
    print("-" * 50)

def sucesso(msg):
    print(f"✅ {msg}")

def erro(msg):
    print(f"❌ {msg}")

def info(msg):
    print(f"ℹ️  {msg}")

def alerta(msg):
    print(f"⚠️  {msg}")

# =========================
# MENU E FLUXOS
# =========================

def exibir_menu():
    print("""
 1  Peers conhecidos
 2  Atualizar peers (rede)
 3  Arquivos locais
 4  Buscar arquivos na rede
 5  Estatísticas
 6  Configurações de download
 ----------------------------
 9  Sair   | q  Quit
""")

def menu_peers(peer):
    titulo("PEERS CONHECIDOS")
    if not peer.vizinhos:
        alerta("Nenhum peer conhecido no momento.")
        pause()
        return

    for i, v in enumerate(peer.vizinhos, 1):
        status = "🟢 ONLINE" if v.status == "ONLINE" else "🔴 OFFLINE"
        print(f"[{i:02d}] {status} | {v.host}:{v.port} | Clock: {v.clock}")

    print("[00] Voltar")
    escolha = input("\n➜ Enviar HELLO para qual peer? ").strip()
    if escolha in ("0", "00", ""): return

    if not escolha.isdigit():
        erro("Digite apenas o número do peer.")
        pause()
        return

    idx = int(escolha) - 1
    if not (0 <= idx < len(peer.vizinhos)):
        erro("Peer inválido.")
        pause()
        return

    alvo = peer.vizinhos[idx]
    info(f"Enviando HELLO para {alvo.host}:{alvo.port}...")
    if peer.enviar_comando(alvo.host, alvo.port, "HELLO"):
        sucesso("HELLO enviado com sucesso.")
    else:
        alerta("Falha ao contatar o peer. Marcado como OFFLINE.")

def menu_arquivos_locais(peer):
    titulo("ARQUIVOS LOCAIS")
    peer.atualizar_meus_arquivos()
    if not peer.meus_arquivos:
        alerta("Nenhum arquivo disponível.")
    else:
        for arq in peer.meus_arquivos:
            print(f"• {arq.nome:<25} {arq.tamanho:>8} bytes")
    pause()

def menu_buscar_arquivos(peer: Peer):
    titulo("BUSCA DE ARQUIVOS NA REDE")
    
    # Limpa buscas anteriores para garantir resultados frescos
    peer.arquivos_rede.clear()
    
    info("Solicitando listas de arquivos para os vizinhos...")
    peer.broadcast_comando("LS")
    
    # Aguarda as respostas chegarem via thread de servidor
    print("⏳ Aguardando respostas dos peers...")
    time.sleep(2) 
    
    if not peer.arquivos_rede:
        alerta("Nenhum arquivo encontrado na rede até o momento.")
        pause()
        return

    # Exibe a tabela formatada conforme o requisito
    print(f"\n{'ID':<4} | {'Nome':<20} | {'Tamanho':<10} | {'Peers'}")
    print("-" * 75)
    print(f"{'[ 0]':<4} | {'':<20} | {'':<10} |") 

    mapeamento_indices = {}
    for i, (nome, dados) in enumerate(peer.arquivos_rede.items(), 1):
        peers_str = ", ".join([f"{p.host}:{p.port}" for p in dados["peers"]])
        print(f"[{i:2d}]  | {nome:<20} | {dados['tamanho']:<10} | {peers_str}")
        mapeamento_indices[i] = nome

    print("-" * 75)
    escolha = input("\n➜ Digite o ID para baixar em paralelo (ou 0 para cancelar): ").strip()

    if escolha.isdigit() and int(escolha) > 0:
        idx = int(escolha)
        if idx in mapeamento_indices:
            nome_arq = mapeamento_indices[idx]
            info_arq = peer.arquivos_rede[nome_arq]
            
            # Inicia a lógica de download paralelo
            peer.solicitar_download(
                nome=nome_arq, 
                tamanho=info_arq["tamanho"], 
                peers_disponiveis=info_arq["peers"]
            )
            sucesso(f"Download de '{nome_arq}' iniciado em paralelo!")
        else:
            erro("ID inválido.")

def menu_chunk(peer):
    titulo("CONFIGURAÇÃO DE DOWNLOAD")
    print(f"Tamanho atual do chunk: {peer.chunk_size} bytes")
    valor = input("Novo tamanho (potência de 2) ou ENTER para cancelar: ").strip()
    if not valor: return
    if not valor.isdigit():
        erro("Valor inválido.")
        return
    n = int(valor)
    if n <= 0 or (n & (n - 1)) != 0:
        erro("O valor deve ser uma potência de 2 (128, 256, 512...).")
    else:
        peer.chunk_size = n
        sucesso(f"Chunk atualizado para {n} bytes.")


# =========================
# MAIN
# =========================

def main():
    # Validação dos argumentos de linha de comando: ./eachare <ip:porta> <vizinhos.txt> <diretorio>
    if len(sys.argv) < 4:
        print("\n" + "!" * 50)
        print(" [ERRO] Argumentos insuficientes.")
        print(" Uso: python main.py <endereco>:<porta> <vizinhos.txt> <diretorio>")
        print(" Ex:  python main.py 127.0.0.1:5000 vizinhos.txt arquivos")
        print("!" * 50 + "\n")
        sys.exit(1)

    try:
        # 1. Processa Endereço e Porta
        addr_port = sys.argv[1]
        if ":" not in addr_port:
            raise ValueError("O endereço deve estar no formato IP:PORTA")
        
        host, port = addr_port.split(":")
        
        # 2. Processa Caminhos persistentes e diretórios
        neighbors_file = sys.argv[2]
        shared_dir = sys.argv[3]

        # Inicializa o Peer (o Servent)
        peer = Peer(
            host=host, 
            port=port, 
            neighbors_path=neighbors_file, 
            files_dir_path=shared_dir
        )
        
        sucesso(f"Peer iniciado em {host}:{port}")
        info(f"Monitorando diretório: {shared_dir}")

        while True:
            limpar_tela()
            header(peer)
            exibir_menu()

            opcao = input("➜ Escolha uma opção: ").strip().lower()

            if opcao == "1":
                menu_peers(peer)

            elif opcao == "2":
                titulo("DESCOBERTA DE PEERS")
                info("Solicitando lista de peers (GET_PEERS) para a rede...")
                peer.broadcast_comando("GET_PEERS")

            elif opcao == "3":
                menu_arquivos_locais(peer)

            elif opcao == "4":
                # Chama a interface de busca agregada e seleção de download
                menu_buscar_arquivos(peer)

            elif opcao == "5":
                titulo("ESTATÍSTICAS DO PEER")
                peer.exibir_estatisticas()
                pause()

            elif opcao == "6":
                menu_chunk(peer)

            elif opcao in ("9", "q", "quit", "sair"):
                titulo("ENCERRANDO")
                peer.broadcast_comando("BYE")
                peer.server.parar() # Encerra o loop do socket server de forma limpa
                sucesso("Peer encerrado com sucesso.")
                break
            else:
                erro("Opção inválida.")
                pause()

    except ValueError as e:
        print(f"\n❌ [ERRO DE ARGUMENTO] {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ [ERRO CRÍTICO] {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()