import unittest
import logging, sys
from unittest.mock import patch, MagicMock

# Configuração básica do logger (pode ser ajustada conforme sua necessidade)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

from modules.client import imprimir_lista_peers

class TestImprimirListaPeers(unittest.TestCase):

    def setUp(self):
        self.vizinhos_dict = {
            "127.0.0.1:9001": {"status": "ONLINE", "clock": 5},
            "192.168.1.10:8000": {"status": "OFFLINE"},
            "10.0.0.5:9005": {"status": "ONLINE", "clock": 10},
            "200.20.20.20:7000": {} # Sem status
        }
        self.vizinhos_vazio = {}

        # Captura a saída do print
        import io
        import sys
        self.held_output = io.StringIO()
        self.old_stdout = sys.stdout
        sys.stdout = self.held_output

    def tearDown(self):
        # Restaura a saída padrão
        sys.stdout = self.old_stdout
        self.held_output.close()

    def get_printed_output(self):
        return self.held_output.getvalue().strip().split('\n')

    def test_lista_com_peers(self):
        imprimir_lista_peers(self.vizinhos_dict)
        output = self.get_printed_output()
        self.assertIn("Lista de peers:", output)
        self.assertIn("\t[0] voltar para o menu anterior", output)
        self.assertIn("\t[1] 127.0.0.1:9001 ONLINE 5", output)
        self.assertIn("\t[2] 192.168.1.10:8000 OFFLINE desconhecido", output)
        self.assertIn("\t[3] 10.0.0.5:9005 ONLINE 10", output)
        self.assertIn("\t[4] 200.20.20.20:7000 desconhecido desconhecido", output)
        logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_lista_com_peers")

    def test_lista_vazia(self):
        imprimir_lista_peers(self.vizinhos_vazio)
        output = self.get_printed_output()
        self.assertIn("Lista de peers:", output)
        self.assertIn("\t[0] voltar para o menu anterior", output)
        self.assertEqual(len(output), 2)
        logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_lista_vazia")

# Suponho que você importe assim:
from modules.client import send_menssage

class TestSendMenssage(unittest.TestCase):

    def log_result(self, test_name, success):
        if success:
            logging.info(f"{Colors.GREEN}[PASS]{Colors.RESET} {test_name}")
        else:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} {test_name}")

    @patch('socket.socket')  # Moca o socket.socket inteiro
    @patch('builtins.print') # Silencia os prints
    def test_envio_mensagem(self, mock_print, mock_socket):
        try:
            # Configura o socket mockado
            mock_client_socket = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_client_socket

            # Definir um peer fictício
            peer = {
                'endereco': '127.0.0.1',
                'porta': '9001',
                'clock': 5
            }
            endereco_servidor = '127.0.0.1'
            porta_servidor = 9002
            tipo = 'HELLO'
            args = ''

            send_menssage(peer, endereco_servidor, porta_servidor, tipo, args)

            # Verifica se conectou no lugar certo
            mock_client_socket.connect.assert_called_once_with((endereco_servidor, porta_servidor))

            # Verifica se mandou a mensagem certa
            mensagem_esperada = f'127.0.0.1:9001 6 HELLO'
            mock_client_socket.sendall.assert_called_once_with(mensagem_esperada.encode())

            self.log_result("test_envio_mensagem", True)

        except AssertionError:
            self.log_result("test_envio_mensagem", False)
            raise

    @patch('socket.socket')
    @patch('builtins.print')
    def test_envio_mensagem_com_args(self, mock_print, mock_socket):
        try:
            mock_client_socket = MagicMock()
            mock_socket.return_value.__enter__.return_value = mock_client_socket

            peer = {
                'endereco': '127.0.0.1',
                'porta': '9001',
                'clock': 7
            }
            endereco_servidor = '192.168.0.10'
            porta_servidor = 8000
            tipo = 'GET_FILE'
            args = 'file.txt'

            send_menssage(peer, endereco_servidor, porta_servidor, tipo, args)

            # Verifica a conexão
            mock_client_socket.connect.assert_called_once_with((endereco_servidor, porta_servidor))

            # Verifica a mensagem enviada
            mensagem_esperada = f'127.0.0.1:9001 8 GET_FILE file.txt'
            mock_client_socket.sendall.assert_called_once_with(mensagem_esperada.encode())

            self.log_result("test_envio_mensagem_com_args", True)

        except AssertionError:
            self.log_result("test_envio_mensagem_com_args", False)
            raise

from modules.client import listar_arquivos_locais

class TestListarArquivosLocais(unittest.TestCase):

    def log_result(self, test_name, success):
        if success:
            logging.info(f"{Colors.GREEN}[PASS]{Colors.RESET} {test_name}")
        else:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} {test_name}")

    @patch('os.listdir')
    @patch('builtins.print')
    def test_listar_arquivos_sucesso(self, mock_print, mock_listdir):
        try:
            # Mockar retorno do listdir
            mock_listdir.return_value = ['arquivo1.txt', 'arquivo2.txt']

            resultado = listar_arquivos_locais('/caminho/falso')

            # Verifica se chamou listdir corretamente
            mock_listdir.assert_called_once_with('/caminho/falso')

            # Verifica se imprimiu os arquivos
            mock_print.assert_any_call('\tarquivo1.txt')
            mock_print.assert_any_call('\tarquivo2.txt')

            # Verifica que a função retorna True
            self.assertTrue(resultado)

            self.log_result("test_listar_arquivos_sucesso", True)

        except AssertionError:
            self.log_result("test_listar_arquivos_sucesso", False)
            raise

    @patch('os.listdir', side_effect=FileNotFoundError("Diretório não encontrado"))
    @patch('builtins.print')
    def test_listar_arquivos_erro(self, mock_print, mock_listdir):
        try:
            resultado = listar_arquivos_locais('/diretorio/inexistente')

            # Verifica que tentou chamar listdir
            mock_listdir.assert_called_once_with('/diretorio/inexistente')

            # Verifica que printou o erro
            mock_print.assert_any_call("Erro listar_arquivos_locais Diretório não encontrado em '/diretorio/inexistente'.")

            # Verifica que a função retorna True mesmo com erro
            self.assertTrue(resultado)

            self.log_result("test_listar_arquivos_erro", True)

        except AssertionError:
            self.log_result("test_listar_arquivos_erro", False)
            raise




if __name__ == '__main__':
    unittest.main(verbosity=1)
