import sys,os
from modules.server import start_server
from modules.client import start_client
from modules.utils import vizinhos_txt_to_dict
import threading

class ValidacaoErro(Exception):
    """
    Exceção personalizada para erros de validação.
    """
    pass

def validar_endereco_porta(endereco_porta):
    """
    Valida se a string está no formato "<endereço>:<porta>" e retorna o endereço e a porta separados.
    """
    try:
        endereco, porta = endereco_porta.split(":")
        porta = int(porta)
        return endereco, porta
    except ValueError:
        raise ValidacaoErro("Endereço e porta devem estar no formato <endereço>:<porta>")

def validar_arquivo(caminho_arquivo):
    """
    Valida se um arquivo existe no caminho especificado.
    """
    if not os.path.isfile(caminho_arquivo):
        raise ValidacaoErro(f"Arquivo {caminho_arquivo} não encontrado.")

def validar_diretorio(caminho_diretorio):
    """
    Valida se um diretório existe no caminho especificado.
    """
    if not os.path.isdir(caminho_diretorio):
        raise ValidacaoErro(f"Diretório {caminho_diretorio} não encontrado.")

def main(endereco_porta, vizinhos_txt, diretorio_compartilhado):
    """
    Função principal que inicia o peer, configurando endereço, porta, lista de vizinhos e diretório compartilhado.
    Inicia threads para o servidor e o cliente do peer.
    """
    # dicionario com dados do peer
    peer = {
        "endereco": "",
        "porta": 0,
        "vizinhos_file": "",
        "vizinhos": [],  
        "diretorio_compartilhado": "",
        "clock": 0
    }
    try:
        endereco, porta = validar_endereco_porta(endereco_porta)
        validar_arquivo(vizinhos_txt)
        validar_diretorio(diretorio_compartilhado)

        peer["endereco"] = endereco
        peer["porta"] = porta
        peer["vizinhos_file"] = vizinhos_txt
        peer["vizinhos"] = vizinhos_txt_to_dict(vizinhos_txt)
        peer["diretorio_compartilhado"] = diretorio_compartilhado

        thread_servidor = threading.Thread(target=start_server, args=(peer,))
        thread_cliente = threading.Thread(target=start_client,args =(peer,))

        thread_servidor.start()
        thread_cliente.start()

        thread_cliente.join()
        thread_servidor.join()

    except ValidacaoErro as e:
        return {"erro": str(e)}

if __name__ == "__main__":
    """
    Ponto de entrada principal do programa.
    Verifica se o número correto de argumentos de linha de comando foi fornecido e chama a função main.
    """
    if len(sys.argv) != 4:
        print("Uso: ./eachare <endereço>:<porta> <vizinhos.txt> <diretório_compartilhado>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])