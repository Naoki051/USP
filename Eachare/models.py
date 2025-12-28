import os
import re

class Mensagem:
    def __init__(self, host, port, clock, command, args=None):
        self.host = host
        self.port = port
        self.clock = clock
        self.command = command
        # Garante que args seja sempre uma lista, mesmo que vazia
        self.args = args if args is not None else []

    def encode(self) -> str:
        """
        Transforma o objeto em string. 
        Ex: ['2', '127.0.0.1:5001', '127.0.0.1:5002'] -> "2 127.0.0.1:5001 127.0.0.1:5002"
        """
        args_str = " ".join(map(str, self.args))
        # Remove espaços extras se não houver argumentos
        corpo = f"{self.host}:{self.port} {self.clock} {self.command} {args_str}"
        return corpo.strip()

    @classmethod
    def decode(cls, data_string: str):
        """
        Decodifica a string separando os 3 primeiros campos e o restante como lista.
        """
        try:
            # Dividimos em no máximo 4 partes: 
            # 0: host:port, 1: clock, 2: command, 3: todo o resto (args)
            partes = data_string.strip().split(' ', 3)
            
            if len(partes) < 3:
                return None

            # 1. Host e Port
            endereco = partes[0].split(':')
            host = endereco[0]
            port = int(endereco[1])
            
            # 2. Clock e Comando
            clock = int(partes[1])
            command = partes[2]
            
            # 3. Argumentos (transforma o resto da string em uma lista)
            args_list = []
            if len(partes) > 3:
                # O split() sem argumentos divide por qualquer quantidade de espaços
                args_list = partes[3].split()
            
            return cls(host, port, clock, command, args_list)
        except Exception as e:
            print(f"Erro ao decodificar mensagem '{data_string}': {e}")
            return None

    def __repr__(self):
        return f"<Msg {self.command} (Args: {len(self.args)}) de {self.host}:{self.port}>"

class Arquivo:
    """Representa um arquivo disponível na rede."""
    def __init__(self, nome, tamanho):
        self.nome = nome
        self.tamanho = tamanho

    def __repr__(self):
        return f"Arquivo(nome='{self.nome}', tamanho={self.tamanho} bytes)"

class Vizinho:
    """Representa um peer remoto conhecido."""
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.status = 'OFFLINE'
        self.clock = 0
        self.arquivos = [] # Lista de objetos Arquivo do vizinho

    @property
    def endereco(self):
        return (self.host, int(self.port))

    def __repr__(self):
        return f"Vizinho({self.host}:{self.port}, status='{self.status}')"
