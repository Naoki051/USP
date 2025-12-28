import os, threading, base64
import time
import math

from models import Mensagem, Arquivo, Vizinho
from client import P2PClient
from server import P2PServer

class Peer:
    def __init__(self, host, port, neighbors_path = "vizinhos.txt", files_dir_path = "arquivos"):
        self.host, self.port = host, int(port)
        self.clock, self.chunk_size = 0, 256
        self.files_dir_path = files_dir_path
        self.neighbors_path = neighbors_path
        
        self.historico_perf = {}
        self.timers_ativos = {} 
        self.arquivos_rede = {}

        self.vizinhos, self.meus_arquivos = [], []
        self.downloads_ativos = {}
        self.lock_download = threading.Lock()

        # Inicia o Servidor (Lado Passivo)
        self.server = P2PServer(self.host, self.port, self.receber_mensagem)
        self.server.iniciar()

        print("\n" + "="*50)
        print(f"🚀 SISTEMA P2P INICIALIZADO | {self.host}:{self.port}")
        print("="*50)
        
        self.carregar_vizinhos()
        self._limpar_arquivos_temporarios()
        self.atualizar_meus_arquivos()

    # =========================================================================
    # COMUNICAÇÃO E SINCRONIZAÇÃO
    # =========================================================================

    def enviar_comando(self, host, port, comando, args=None):
        """Interface de saída: Lado Ativo (Cliente)."""
        self.clock += 1
        msg = Mensagem(self.host, self.port, self.clock, comando, args)
        print(f"  >>> [ENVIANDO] '{msg.encode()}' Para: {host}:{port} |")
        if not P2PClient.enviar(host, port, msg):
            print(f"  !!! [FALHA] Não foi possível contatar {host}:{port}. Marcando como OFFLINE.")
            self.atualiza_vizinho_status(host, port, 0, 'OFFLINE')
    
    def broadcast_comando(self, comando, args=None):
        """Envia um comando para todos os vizinhos que não estão OFFLINE."""
        for v in self.vizinhos:
            if v.status != 'OFFLINE' or comando == "BYE" or comando == "GET_PEERS":
                self.enviar_comando(v.host, v.port, comando, args)

    def receber_mensagem(self, msg):
        """Interface de entrada: Ponto de acesso do Servidor."""
        velho_clock = self.clock
        self.clock = max(self.clock, msg.clock) + 1
        
        print(f"\n  <<< [RECEBIDO] {msg.encode()}")
        print(f"      [CLOCK] Sincronia: Local({velho_clock}) + Remoto({msg.clock}) ➔ Novo({self.clock})")
        
        self.atualiza_vizinho_status(msg.host, msg.port, msg.clock, 'ONLINE')
        self.processar_comando(msg)

    def processar_comando(self, msg):
        """Roteador de lógica de comandos."""
        if msg.command == "HELLO":
            self.enviar_comando(msg.host, msg.port, "HELLO_RESPONSE")
        
        elif msg.command == "GET_PEERS":
            vizinhos_filtrados = [
                f"{v.host}:{v.port}:{v.status}:{v.clock}" 
                for v in self.vizinhos 
                if not (v.host == msg.host and v.port == int(msg.port))
            ]
            args_res = [len(vizinhos_filtrados)] + vizinhos_filtrados
            self.enviar_comando(msg.host, msg.port, "PEER_LIST", args_res)

        elif msg.command == "PEER_LIST":
            if not msg.args: return
            for info in msg.args[1:]:
                try:
                    p_host, p_port, p_status, p_clock = info.split(':')
                    if p_host == self.host and int(p_port) == self.port:
                        continue
                    self.atualiza_vizinho_status(p_host, p_port, p_clock, p_status)
                except (ValueError, IndexError):
                    continue

        elif msg.command == "LS":
            lista = [f"{a.nome}:{a.tamanho}" for a in self.meus_arquivos]
            self.enviar_comando(msg.host, msg.port, "LS_LIST", [len(lista)] + lista)
        elif msg.command == "LS_LIST":
            # Recebe a lista de um vizinho: [quantidade, "arq1:tam1", "arq2:tam2"]
            for info in msg.args[1:]:
                try:
                    nome, tam = info.split(':')
                    tam = int(tam)
                    with self.lock_download: # Usamos o lock para evitar concorrência na lista
                        if nome not in self.arquivos_rede:
                            self.arquivos_rede[nome] = {"tamanho": tam, "peers": []}
                        # Adiciona o peer à lista de donos se ele ainda não estiver lá
                        vizinho_dono = Vizinho(msg.host, msg.port)
                        if not any(p.host == msg.host and p.port == msg.port for p in self.arquivos_rede[nome]["peers"]):
                            self.arquivos_rede[nome]["peers"].append(vizinho_dono)
                except ValueError:
                    continue
        
        elif msg.command == "DL":
            self._servir_chunk(msg)
            
        elif msg.command == "FILE":
            self._gravar_chunk(msg)
            
        elif msg.command == "BYE":
            print(f"  [AVISO] O vizinho {msg.host}:{msg.port} saiu da rede.")
            self.atualiza_vizinho_status(msg.host, msg.port, msg.clock, 'OFFLINE')

    # =========================================================================
    # TRANSFERÊNCIA DE ARQUIVOS (PARALELA)
    # =========================================================================

    def solicitar_download(self, nome, tamanho, peers_disponiveis):
        num_peers = len(peers_disponiveis)
        num_chunks = (int(tamanho) + self.chunk_size - 1) // self.chunk_size
        caminho = os.path.join(self.files_dir_path, "downloading_" + nome)

        print("\n" + "-"*50)
        print(f"📂 INICIANDO TRANSFERÊNCIA: {nome}")
        print(f"   Tamanho: {tamanho} bytes | Chunks: {num_chunks} | Peers: {num_peers}")
        print("-"*50)

        self.timers_ativos[nome] = {
            "start": time.perf_counter(),
            "peers": num_peers,
            "size": tamanho,
            "chunk_size": self.chunk_size
        }
        
        with self.lock_download:
            with open(caminho, "wb") as f: f.truncate(int(tamanho))
            self.downloads_ativos[nome] = {
                "total": num_chunks, "recebidos": set(), "lock": threading.Lock()
            }

        for i in range(1, num_chunks + 1):
            target = peers_disponiveis[(i-1) % len(peers_disponiveis)]
            self.enviar_comando(target.host, target.port, "DL", [nome, self.chunk_size, i])

    def _gravar_chunk(self, msg):
        nome, size, idx, b64 = msg.args
        idx, size = int(idx), int(size)
        
        with self.lock_download:
            if nome not in self.downloads_ativos: return
            ctx = self.downloads_ativos[nome]

        with ctx["lock"]:
            path = os.path.join(self.files_dir_path, "downloading_" + nome)
            with open(path, "r+b") as f:
                f.seek((idx-1) * size)
                f.write(base64.b64decode(b64))
            
            ctx["recebidos"].add(idx)
            percent = (len(ctx["recebidos"]) / ctx["total"]) * 100
            print(f"  [PROGRESSO] {nome}: Chunk #{idx} recebido ({percent:.1f}%)")
            
            if len(ctx["recebidos"]) >= ctx["total"]:
                self._finalizar(nome)

    def _servir_chunk(self, msg):
        nome, size, idx = msg.args
        idx, size = int(idx), int(size)
        path = os.path.join(self.files_dir_path, nome)
        if os.path.exists(path):
            with open(path, "rb") as f:
                f.seek((idx-1) * size)
                dados = f.read(size)
                b64 = base64.b64encode(dados).decode()
                self.enviar_comando(msg.host, msg.port, "FILE", [nome, size, idx, b64])

    def _finalizar(self, nome):
        end_time = time.perf_counter()
        
        # 1. Recupera e remove o timer e o estado de download ativo
        # Isso evita que o arquivo apareça como "em andamento" nas estatísticas
        dados = self.timers_ativos.pop(nome, None)
        with self.lock_download:
            if nome in self.downloads_ativos:
                del self.downloads_ativos[nome]

        if dados:
            tempo_total = end_time - dados["start"]
            tripla = (dados["chunk_size"], dados["peers"], dados["size"])
            if tripla not in self.historico_perf:
                self.historico_perf[tripla] = []
            self.historico_perf[tripla].append(tempo_total)

        temp_path = os.path.join(self.files_dir_path, "downloading_" + nome)
        final_path = os.path.join(self.files_dir_path, nome)

        # 2. Renomeia o arquivo (os.replace sobrescreve se já existir, evitando erros)
        try:
            if os.path.exists(temp_path):
                os.replace(temp_path, final_path)
                print(f"\n✅ DOWNLOAD CONCLUÍDO E ARQUIVO RENOMEADO: {nome}")
        except Exception as e:
            print(f"❌ Erro ao renomear arquivo: {e}")
        
        # 3. Atualiza a lista local (que agora deve filtrar os temporários)
        self.atualizar_meus_arquivos()

    def _limpar_arquivos_temporarios(self):
        """Remove resíduos de downloads que não foram finalizados em sessões anteriores."""
        print(f"  [SISTEMA] Verificando resíduos em '{self.files_dir_path}'...")
        
        if not os.path.exists(self.files_dir_path):
            return

        removidos = 0
        for arquivo in os.listdir(self.files_dir_path):
            if arquivo.startswith("downloading_"):
                try:
                    caminho_completo = os.path.join(self.files_dir_path, arquivo)
                    os.remove(caminho_completo)
                    removidos += 1
                except Exception as e:
                    print(f"  !!! [ERRO] Não foi possível remover o arquivo temporário {arquivo}: {e}")
        
        if removidos > 0:
            print(f"  [OK] Limpeza concluída: {removidos} arquivo(s) temporário(s) removido(s).")
        else:
            print("  [OK] Nenhum arquivo temporário encontrado.")
    
    def exibir_estatisticas(self):
        print("\n" + "-"*65)
        print(f"{'Tam. chunk':<12} | {'N peers':<8} | {'Tam. arquivo':<12} | {'N':<3} | {'Tempo [s]':<10} | {'Desvio':<8}")
        print("-"*65)

        for tripla, tempos in self.historico_perf.items():
            chunk_z, n_peers, file_z = tripla
            n = len(tempos)
            
            # Cálculo da Média
            media = sum(tempos) / n
            
            # Cálculo do Desvio Padrão
            if n > 1:
                variancia = sum((x - media) ** 2 for x in tempos) / n
                desvio = math.sqrt(variancia)
            else:
                desvio = 0.0

            # Formatação conforme o exemplo
            print(f"{chunk_z:<12} | {n_peers:<8} | {file_z:<12} | {n:<3} | {media:<10.5f} | {desvio:<8.5f}")
        
        if not self.historico_perf:
            print("Nenhum dado de performance registrado ainda.")
        print("-"*65)


    # =========================================================================
    # GESTÃO DE ESTADO E VIZINHOS
    # =========================================================================

    def atualiza_vizinho_status(self, host, port, clock, status):
        clock, port = int(clock), int(port)
        
        for v in self.vizinhos:
            if v.host == host and v.port == port:
                if clock > v.clock:
                    if v.status != status:
                        print(f"  [STATUS] {host}:{port} está agora {status}")
                    v.clock, v.status = clock, status
                return

        print(f"  [SISTEMA] Novo vizinho descoberto: {host}:{port}")
        novo = Vizinho(host, port)
        novo.clock, novo.status = clock, status
        self.vizinhos.append(novo)
        
        with open(self.neighbors_path, 'a') as f:
            f.write(f"{host}:{port}\n")

    def carregar_vizinhos(self):
        print(f"  [ARQUIVO] Carregando vizinhos de {self.neighbors_path}...")
        if not os.path.exists(self.neighbors_path): return
        with open(self.neighbors_path, 'r') as f:
            for l in f:
                try:
                    h, p = l.strip().split(':')
                    if not any(v.host == h and v.port == int(p) for v in self.vizinhos):
                        self.vizinhos.append(Vizinho(h, int(p)))
                except: continue

    def atualizar_meus_arquivos(self):
        os.makedirs(self.files_dir_path, exist_ok=True)
        self.meus_arquivos = [Arquivo(f, os.path.getsize(os.path.join(self.files_dir_path, f))) 
                              for f in os.listdir(self.files_dir_path) if os.path.isfile(os.path.join(self.files_dir_path, f))]

    def encerrar_nó(self):
        print("\n[SISTEMA] Encerrando serviços...")
        self._limpar_arquivos_temporarios()
        self.server.parar()