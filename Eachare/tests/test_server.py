import unittest
from unittest.mock import patch, mock_open
import logging
import base64

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    CYAN = '\033[96m'
    RESET = '\033[0m'

# Supondo que importe assim:
from modules.server import construir_resposta_peers, construir_resposta_ls, construir_resposta_dl

class TestConstruirRespostas(unittest.TestCase):

    def test_construir_resposta_peers(self):
        peer = {
            'endereco': '127.0.0.1',
            'porta': 8000,
            'clock': 5,
            'vizinhos': {
                '127.0.0.2:8001': {'status': 'ONLINE', 'clock': 2},
                '127.0.0.3:8002': {'status': 'OFFLINE', 'clock': 1},
                '127.0.0.4:8003': {'status': 'ONLINE', 'clock': 3},
            }
        }
        mensagem_separada = {
            'endereco_origem': '127.0.0.2',
            'porta_origem': 8001
        }

        resposta = construir_resposta_peers(peer, mensagem_separada)

        self.assertIn('127.0.0.3:8002:OFFLINE:1', resposta)
        self.assertIn('127.0.0.4:8003:ONLINE:3', resposta)
        self.assertNotIn('127.0.0.2:8001', resposta)
        self.assertIn('PEER_LIST 2', resposta)
        logger.info(f"{Colors.GREEN}[✔ SUCCESS]{Colors.RESET} test_construir_resposta_peers")

    @patch('os.listdir')
    @patch('os.path.isfile', return_value=True)
    @patch('os.path.getsize')
    def test_construir_resposta_ls(self, mock_getsize, mock_isfile, mock_listdir):
        peer = {
            'endereco': '127.0.0.1',
            'porta': 8000,
            'clock': 7,
            'diretorio_compartilhado': '/fake/diretorio'
        }

        mock_listdir.return_value = ['file1.txt', 'file2.txt']
        mock_getsize.side_effect = [123, 456]

        resposta = construir_resposta_ls(peer)

        self.assertIn('file1.txt:123', resposta)
        self.assertIn('file2.txt:456', resposta)
        self.assertIn('LS_LIST 2', resposta)
        logger.info(f"{Colors.GREEN}[✔ SUCCESS]{Colors.RESET} test_construir_resposta_ls")

    @patch('builtins.open', new_callable=mock_open, read_data=b'conteudo de teste')
    @patch('os.path.join', return_value='arquivos/file1.txt')
    def test_construir_resposta_dl(self, mock_path_join, mock_file):
        peer = {
            'endereco': '127.0.0.1',
            'porta': 8000,
            'clock': 10
        }

        resposta = construir_resposta_dl(peer, 'file1.txt')

        conteudo_base64 = base64.b64encode(b'conteudo de teste').decode('utf-8')
        self.assertIsNotNone(resposta)
        self.assertIn('FILE', resposta)
        self.assertIn('file1.txt', resposta)
        self.assertIn(conteudo_base64, resposta)
        logger.info(f"{Colors.GREEN}[✔ SUCCESS]{Colors.RESET} test_construir_resposta_dl")

    @patch('builtins.open', side_effect=FileNotFoundError)
    @patch('os.path.join', return_value='arquivos/file_inexistente.txt')
    def test_construir_resposta_dl_arquivo_nao_encontrado(self, mock_path_join, mock_file):
        peer = {
            'endereco': '127.0.0.1',
            'porta': 8000,
            'clock': 10
        }

        resposta = construir_resposta_dl(peer, 'file_inexistente.txt')
        self.assertIsNone(resposta)
        logger.info(f"{Colors.GREEN}[✔ SUCCESS]{Colors.RESET} test_construir_resposta_dl_arquivo_nao_encontrado")
if __name__ == "__main__":
    unittest.main()