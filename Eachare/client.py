import socket
from models import Mensagem

class P2PClient:
    @staticmethod
    def enviar(host, port, mensagem: Mensagem):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(3)
                s.connect((host, int(port)))
                s.sendall(mensagem.encode().encode('utf-8'))
                return True
        except: return False