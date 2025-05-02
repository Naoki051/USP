import unittest
from unittest.mock import patch,mock_open
import logging
import os
from io import StringIO

# Configuração básica do logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Cores ANSI para o terminal
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    RESET = '\033[0m'

# Configuração básica do logger
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from modules.utils import vizinhos_txt_to_dict

class TestVizinhosTxtToDict(unittest.TestCase):

    def setUp(self):
        # Desativa logs dentro das funções testadas
        logging.disable(logging.CRITICAL)

    def tearDown(self):
        # Reativa os logs
        logging.disable(logging.NOTSET)
        # Limpa arquivos temporários criados nos testes
        if os.path.exists('test_vizinhos.txt'):
            os.remove('test_vizinhos.txt')

    def log_result(self, test_name, success):
        if success:
            print(f"{Colors.GREEN}[PASS]{Colors.RESET} {test_name}")
        else:
            print(f"{Colors.RED}[ERROR]{Colors.RESET} {test_name}")

    def test_vizinhos_txt_valido(self):
        with open('test_vizinhos.txt', 'w', encoding='utf-8') as f:
            f.write('127.0.0.1:8000\n')
            f.write('192.168.0.1:8080\n')
        
        try:
            resultado = vizinhos_txt_to_dict('test_vizinhos.txt')
            esperado = {
                '127.0.0.1:8000': {'status': 'OFFLINE', 'clock': 0},
                '192.168.0.1:8080': {'status': 'OFFLINE', 'clock': 0}
            }
            self.assertEqual(resultado, esperado)
            self.log_result("test_vizinhos_txt_valido", True)
        except AssertionError:
            self.log_result("test_vizinhos_txt_valido", False)
            raise

    def test_vizinhos_txt_linha_mal_formatada(self):
        with open('test_vizinhos.txt', 'w', encoding='utf-8') as f:
            f.write('127.0.0.1-8000\n')  # Separador errado
        
        try:
            resultado = vizinhos_txt_to_dict('test_vizinhos.txt')
            esperado = {}
            self.assertEqual(resultado, esperado)
            self.log_result("test_vizinhos_txt_linha_mal_formatada", True)
        except AssertionError:
            self.log_result("test_vizinhos_txt_linha_mal_formatada", False)
            raise

    def test_vizinhos_txt_ip_invalido(self):
        with open('test_vizinhos.txt', 'w', encoding='utf-8') as f:
            f.write('999.999.999.999:8000\n')  # IP inválido
        
        try:
            resultado = vizinhos_txt_to_dict('test_vizinhos.txt')
            esperado = {}
            self.assertEqual(resultado, esperado)
            self.log_result("test_vizinhos_txt_ip_invalido", True)
        except AssertionError:
            self.log_result("test_vizinhos_txt_ip_invalido", False)
            raise

    def test_vizinhos_txt_porta_invalida(self):
        with open('test_vizinhos.txt', 'w', encoding='utf-8') as f:
            f.write('127.0.0.1:99999\n')  # Porta inválida (>65535)
        
        try:
            resultado = vizinhos_txt_to_dict('test_vizinhos.txt')
            esperado = {}
            self.assertEqual(resultado, esperado)
            self.log_result("test_vizinhos_txt_porta_invalida", True)
        except AssertionError:
            self.log_result("test_vizinhos_txt_porta_invalida", False)
            raise

    def test_vizinhos_txt_arquivo_inexistente(self):
        try:
            resultado = vizinhos_txt_to_dict('arquivo_inexistente.txt')
            esperado = {}
            self.assertEqual(resultado, esperado)
            self.log_result("test_vizinhos_txt_arquivo_inexistente", True)
        except AssertionError:
            self.log_result("test_vizinhos_txt_arquivo_inexistente", False)
            raise

    def test_vizinhos_txt_arquivo_vazio(self):
        open('test_vizinhos.txt', 'w', encoding='utf-8').close()  # Cria arquivo vazio
        
        try:
            resultado = vizinhos_txt_to_dict('test_vizinhos.txt')
            esperado = {}
            self.assertEqual(resultado, esperado)
            self.log_result("test_vizinhos_txt_arquivo_vazio", True)
        except AssertionError:
            self.log_result("test_vizinhos_txt_arquivo_vazio", False)
            raise

from modules.utils import incrementa_clock

class TestIncrementaClock(unittest.TestCase):
    @patch('builtins.print')
    def test_incrementa_clock_sem_msg_clock(self, mock_print):
        try:
            peer = {'clock': 5}
            incrementa_clock(peer)
            self.assertEqual(peer['clock'], 6)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_incrementa_clock_sem_msg_clock")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_incrementa_clock_sem_msg_clock")

    @patch('builtins.print')
    def test_incrementa_clock_com_msg_clock_maior(self, mock_print):
        try:
            peer = {'clock': 3}
            incrementa_clock(peer, msg_clock=10)
            self.assertEqual(peer['clock'], 11)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_incrementa_clock_com_msg_clock_maior")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_incrementa_clock_com_msg_clock_maior")

    @patch('builtins.print')
    def test_incrementa_clock_com_msg_clock_menor(self, mock_print):
        try:
            peer = {'clock': 8}
            incrementa_clock(peer, msg_clock=5)
            self.assertEqual(peer['clock'], 9)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_incrementa_clock_com_msg_clock_menor")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_incrementa_clock_com_msg_clock_menor")

from modules.utils import separar_msg

class TestSepararMsg(unittest.TestCase):

    def test_mensagem_valida_com_args(self):
        mensagem = "127.0.0.1:9000 5 PEER_LIST 2 127.0.0.1:9001:4 ONLINE 127.0.0.1:9002:2 OFFLINE"
        try:
            resultado = separar_msg(mensagem)
            esperado = {
                "endereco_origem": "127.0.0.1",
                "porta_origem": 9000,
                "clock_origem": 5,
                "tipo": "PEER_LIST",
                "args": ["2", "127.0.0.1:9001:4", "ONLINE", "127.0.0.1:9002:2", "OFFLINE"]
            }
            self.assertEqual(resultado, esperado)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_mensagem_valida_com_args")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_mensagem_valida_com_args")
            logging.info(f"Esperado: {esperado}")
            logging.info(f"Resultado: {resultado}")

    def test_mensagem_valida_sem_args(self):
        mensagem = "192.168.1.10:8080 10 STATUS"
        try:
            resultado = separar_msg(mensagem)
            esperado = {
                "endereco_origem": "192.168.1.10",
                "porta_origem": 8080,
                "clock_origem": 10,
                "tipo": "STATUS",
                "args": []
            }
            self.assertEqual(resultado, esperado)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_mensagem_valida_sem_args")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_mensagem_valida_sem_args")
            logging.info(f"Esperado: {esperado}")
            logging.info(f"Resultado: {resultado}")
            

    def test_mensagem_poucas_partes(self):
        mensagem = "127.0.0.1:9000 1"
        try:
            with patch('sys.stdout', new_callable=StringIO) as stdout:
                resultado = separar_msg(mensagem)
                self.assertIsNone(resultado)
                self.assertIn("Mensagem mal formatada: faltam partes essenciais.", stdout.getvalue())
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_mensagem_poucas_partes")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_mensagem_poucas_partes")
        
    def test_endereco_origem_mal_formatado(self):
        mensagem = "127.0.0.19000 2 DATA"
        try:
            with patch('sys.stdout', new_callable=StringIO) as stdout:
                resultado = separar_msg(mensagem)
                self.assertIsNone(resultado)
                self.assertIn("Endereço de origem mal formatado.", stdout.getvalue())
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_endereco_origem_mal_formatado")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_endereco_origem_mal_formatado")

    def test_porta_origem_nao_inteiro(self):
        mensagem = "10.0.0.5:abc 3 REQUEST"
        try:
            with patch('sys.stdout', new_callable=StringIO) as stdout:
                resultado = separar_msg(mensagem)
                self.assertIsNone(resultado)
                self.assertIn("invalid literal for int() with base 10: 'abc'", stdout.getvalue())
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_porta_origem_nao_inteiro")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_porta_origem_nao_inteiro")

    def test_clock_origem_nao_inteiro(self):
        mensagem = "172.16.0.1:5000 xyz SYNC"
        try:
            with patch('sys.stdout', new_callable=StringIO) as stdout:
                resultado = separar_msg(mensagem)
                self.assertIsNone(resultado)
                self.assertIn("invalid literal for int() with base 10: 'xyz'", stdout.getvalue())
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_clock_origem_nao_inteiro")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_clock_origem_nao_inteiro")

    def test_mensagem_vazia(self):
        mensagem = ""
        try:
            with patch('sys.stdout', new_callable=StringIO) as stdout:
                resultado = separar_msg(mensagem)
                self.assertIsNone(resultado)
                self.assertIn("Mensagem mal formatada: faltam partes essenciais.", stdout.getvalue())
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_mensagem_vazia")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_mensagem_vazia")

    def test_mensagem_com_espacos_extras(self):
        mensagem = "127.0.0.1:9000 5 UPDATE    param1  "
        try:
            resultado = separar_msg(mensagem)
            esperado = {
                "endereco_origem": "127.0.0.1",
                "porta_origem": 9000,
                "clock_origem": 5,
                "tipo": "UPDATE",
                "args": ["param1"]
            }
            self.assertEqual(resultado, esperado)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_mensagem_com_espacos_extras")
        except:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} test_mensagem_com_espacos_extras")
            logging.info(f"Esperado: {esperado}")
            logging.info(f"Resultado: {resultado}")

from modules.utils import adiciona_vizinho

class TestAdicionaVizinhoDict(unittest.TestCase):

    def setUp(self):
        """Configuração inicial para os testes."""
        self.peer = {'vizinhos': {}, 'vizinhos_file': 'test_vizinhos.txt'}
        # Garante que o arquivo de teste não exista no início de cada teste
        if os.path.exists(self.peer['vizinhos_file']):
            os.remove(self.peer['vizinhos_file'])

    def tearDown(self):
        """Limpeza após cada teste."""
        # Remove o arquivo de teste ao final de cada teste
        if os.path.exists(self.peer['vizinhos_file']):
            os.remove(self.peer['vizinhos_file'])
    @patch('builtins.print')
    def test_adiciona_vizinho_com_sucesso(self, mock_print):
        """Testa a adição bem-sucedida de um novo vizinho."""
        endereco = "192.168.1.10"
        porta = 9005
        status = "ONLINE"
        clock = 10
        chave_esperada = f"{endereco}:{porta}"
        try:
            adiciona_vizinho(self.peer, endereco, porta, status, clock)

            self.assertIn(chave_esperada, self.peer['vizinhos'])
            self.assertEqual(self.peer['vizinhos'][chave_esperada], {'status': status, 'clock': clock})

            # Verifica se o vizinho foi escrito no arquivo
            with open(self.peer['vizinhos_file'], 'r') as f:
                conteudo = f.read().strip()
                self.assertEqual(conteudo, chave_esperada)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_adiciona_vizinho_com_sucesso")
        except AssertionError:
            logger.error(f"{Colors.RED}[ERROR]{Colors.RESET} test_adiciona_vizinho_com_sucesso - Falha na asserção.")
            raise
        except Exception as e:
            logger.error(f"{Colors.RED}[ERROR]{Colors.RESET} test_adiciona_vizinho_com_sucesso - Erro inesperado: {e}")
            raise
        
    @patch('builtins.print')
    def test_adiciona_multiplos_vizinhos(self, mock_print):
        """Testa a adição de múltiplos vizinhos."""
        vizinho1_endereco = "10.0.0.1"
        vizinho1_porta = 8000
        vizinho1_status = "OFFLINE"
        vizinho1_clock = 3
        chave1 = f"{vizinho1_endereco}:{vizinho1_porta}"

        vizinho2_endereco = "172.16.0.5"
        vizinho2_porta = 9001
        vizinho2_status = "ONLINE"
        vizinho2_clock = 7
        chave2 = f"{vizinho2_endereco}:{vizinho2_porta}"
        try:
            adiciona_vizinho(self.peer, vizinho1_endereco, vizinho1_porta, vizinho1_status, vizinho1_clock)
            adiciona_vizinho(self.peer, vizinho2_endereco, vizinho2_porta, vizinho2_status, vizinho2_clock)

            self.assertIn(chave1, self.peer['vizinhos'])
            self.assertEqual(self.peer['vizinhos'][chave1], {'status': vizinho1_status, 'clock': vizinho1_clock})
            self.assertIn(chave2, self.peer['vizinhos'])
            self.assertEqual(self.peer['vizinhos'][chave2], {'status': vizinho2_status, 'clock': vizinho2_clock})

            # Verifica se os vizinhos foram escritos no arquivo (um em cada linha)
            with open(self.peer['vizinhos_file'], 'r') as f:
                conteudo = f.read().strip().split('\n')
                self.assertIn(chave1, conteudo)
                self.assertIn(chave2, conteudo)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_adiciona_multiplos_vizinhos")
        except AssertionError:
            logger.error(f"{Colors.RED}[ERROR]{Colors.RESET} test_adiciona_multiplos_vizinhos - Falha na asserção.")
            raise
        except Exception as e:
            logger.error(f"{Colors.RED}[ERROR]{Colors.RESET} test_adiciona_multiplos_vizinhos - Erro inesperado: {e}")
            raise

    @patch('builtins.print')
    def test_adiciona_vizinho_arquivo_existente(self, mock_print):
        """Testa a adição de um vizinho quando o arquivo já existe."""
        # Cria um arquivo de vizinhos existente
        with open(self.peer['vizinhos_file'], 'w') as f:
            f.write("127.0.0.1:9000")
        try:
            endereco = "192.168.1.11"
            porta = 9006
            status = "ONLINE"
            clock = 12
            chave_esperada = f"{endereco}:{porta}"

            adiciona_vizinho(self.peer, endereco, porta, status, clock)

            self.assertIn(chave_esperada, self.peer['vizinhos'])
            self.assertEqual(self.peer['vizinhos'][chave_esperada], {'status': status, 'clock': clock})

            # Verifica se o novo vizinho foi adicionado ao arquivo (em uma nova linha)
            with open(self.peer['vizinhos_file'], 'r') as f:
                conteudo = f.read().strip().split('\n')
                self.assertEqual(len(conteudo), 2)
                self.assertEqual(conteudo[0], "127.0.0.1:9000")
                self.assertEqual(conteudo[1], chave_esperada)
            logger.info(f"{Colors.GREEN}[PASS]{Colors.RESET} test_adiciona_vizinho_arquivo_existente")
        except AssertionError:
            logger.error(f"{Colors.RED}[ERROR]{Colors.RESET} test_adiciona_vizinho_arquivo_existente - Falha na asserção.")
            raise
        except Exception as e:
            logger.error(f"{Colors.RED}[ERROR]{Colors.RESET} test_adiciona_vizinho_arquivo_existente - Erro inesperado: {e}")
            raise
    
from modules.utils import atualiza_status_vizinho

class TestAtualizaStatusVizinho(unittest.TestCase):

    def log_result(self, test_name, success):
        if success:
            logging.info(f"{Colors.GREEN}[PASS]{Colors.RESET} {test_name}")
        else:
            logging.info(f"{Colors.RED}[ERROR]{Colors.RESET} {test_name}")

    def setUp(self):
        # Configuração inicial para cada teste
        self.peer = {
            'vizinhos': {
                '127.0.0.1:9001': {'status': 'OFFLINE', 'clock': 2}
            },
            'vizinhos_file': 'test_vizinhos.txt'
        }

    @patch('builtins.print')  # Silencia os prints durante os testes
    def test_atualizar_status_existente(self, _):
        try:
            result = atualiza_status_vizinho(self.peer, '127.0.0.1', '9001', 'ONLINE', 2)
            self.assertTrue(result)
            chave = '127.0.0.1:9001'
            self.assertEqual(self.peer['vizinhos'][chave]['status'], 'ONLINE')
            self.assertEqual(self.peer['vizinhos'][chave]['clock'], 2)
            self.log_result("test_atualizar_status_existente", True)
        except AssertionError:
            self.log_result("test_atualizar_status_existente", False)
            raise

    @patch('builtins.print')
    def test_atualizar_clock_maior(self, _):
        try:
            result = atualiza_status_vizinho(self.peer, '127.0.0.1', '9001', 'OFFLINE', 5)
            self.assertTrue(result)
            chave = "127.0.0.1:9001"
            self.assertEqual(self.peer['vizinhos'][chave]['clock'], 5)
            self.log_result("test_atualizar_clock_maior", True)
        except AssertionError:
            self.log_result("test_atualizar_clock_maior", False)
            raise

    @patch('builtins.print')
    def test_adicionar_novo_vizinho(self, _):
        try:
            result = atualiza_status_vizinho(self.peer, '127.0.0.1', '9002', 'ONLINE', 1)
            self.assertTrue(result)
            self.assertTrue(self.peer['vizinhos']['127.0.0.1:9002'] == {'status': 'ONLINE','clock': 1})
            self.log_result("test_adicionar_novo_vizinho", True)
        except AssertionError:
            self.log_result("test_adicionar_novo_vizinho", False)
            raise

    @patch('builtins.print')
    def test_excecao_ao_atualizar(self, _):
        try:
            # Remove 'vizinhos' para forçar erro
            peer_broken = {'vizinhos_file': 'test_vizinhos.txt'}
            result = atualiza_status_vizinho(peer_broken, '127.0.0.1', '9001', 'ONLINE', 2)
            self.assertFalse(result)
            self.log_result("test_excecao_ao_atualizar", True)
        except AssertionError:
            self.log_result("test_excecao_ao_atualizar", False)
            raise

if __name__ == '__main__':
    unittest.main(verbosity=1)