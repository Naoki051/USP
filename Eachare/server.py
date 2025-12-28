import socket
import threading
from models import Mensagem

class P2PServer:
    def __init__(self, host, port, callback_processar):
        self.host = host
        self.port = port
        self.callback = callback_processar
        self.ativo = True  # Flag de controle

    def iniciar(self):
        # Mantemos daemon=True como garantia extra
        self.thread_principal = threading.Thread(target=self._rodar, daemon=True)
        self.thread_principal.start()

    def parar(self):
        """Metodo para interromper o loop do servidor."""
        self.ativo = False
        print(f"[SERVIDOR] Encerrando escuta em {self.host}:{self.port}...")

    def _rodar(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # Define um timeout de 1 segundo para o socket
            # Isso faz o s.accept() "acordar" a cada 1 segundo se ninguem conectar
            s.settimeout(1.0) 
            s.bind((self.host, self.port))
            s.listen()

            while self.ativo:
                try:
                    conn, addr = s.accept()
                    threading.Thread(target=self._lidar, args=(conn,), daemon=True).start()
                except socket.timeout:
                    # Se der timeout, o loop volta ao inicio e checa 'self.ativo'
                    continue
                except Exception as e:
                    if self.ativo:
                        print(f"[ERRO SERVIDOR] {e}")
            
        print("[SERVIDOR] Socket fechado com sucesso.")

    def _lidar(self, conn):
        with conn:
            try:
                # Aumentamos um pouco o buffer para suportar mensagens maiores com Base64
                dados = conn.recv(10240).decode('utf-8')
                if dados:
                    msg = Mensagem.decode(dados)
                    if msg: 
                        self.callback(msg)
            except Exception as e:
                pass