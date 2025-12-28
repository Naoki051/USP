Reconhecimento de Dígitos com CNN (MNIST)
Visão Geral do Projeto
Este projeto consiste em uma implementação de uma Rede Neural Convolucional (CNN) em Python para a tarefa de reconhecimento de dígitos manuscritos. Utilizando o dataset MNIST, o script é capaz de treinar, avaliar e gerar artefatos para duas tarefas distintas:

Classificação Multiclasse: Reconhecimento de todos os 10 dígitos (0-9).

Classificação Binária: Distinção entre dois dígitos específicos, configuráveis pelo usuário.

O script é altamente flexível e foi projetado para facilitar a experimentação, permitindo o ajuste de diversos hiperparâmetros via linha de comando e salvando todos os resultados de forma organizada para análise e reprodutibilidade.

Principais Funcionalidades
Modelo CNN Flexível: Arquitetura com número de camadas convolucionais configurável.

Dupla Funcionalidade: Suporte nativo para tarefas de classificação multiclass e binary.

Regularização com Dropout: Inclui a camada de Dropout para mitigar o overfitting.

Geração Automática de Gráficos: Cria e salva visualizações importantes como curvas de aprendizado, matriz de confusão e curva ROC.

Geração de Artefatos: Salva todos os arquivos essenciais de cada execução: hiperparâmetros, pesos do modelo (iniciais e finais), logs de treinamento e previsões.

Controle via Linha de Comando: Utiliza argparse para uma interface de usuário limpa e fácil de usar.

Medição de Desempenho: Calcula e salva o tempo gasto no treinamento.

Tecnologias Utilizadas
Python 3.9+

TensorFlow 2.x (com Keras)

Scikit-learn

NumPy

Pandas

Matplotlib & Seaborn

Instalação e Configuração do Ambiente
Para executar este projeto, siga os passos abaixo:

Clone o repositório:

git clone https://github.com/seu-usuario/seu-repositorio.git
cd seu-repositorio

Crie e ative um ambiente virtual (recomendado):

# Cria o ambiente

python -m venv venv

# Ativa o ambiente
# No Windows:
.\venv\Scripts\activate
# No macOS/Linux:
source venv/bin/activate

Instale as dependências:
O arquivo requirements.txt contém todas as bibliotecas necessárias.

pip install -r requirements.txt

Como Usar o Projeto
O script principal é o run_experiment.py. Todos os parâmetros podem ser ajustados via linha de comando.

Exemplos de Execução

1. Executar a Tarefa Multiclasse (Padrão):
Este comando treinará o modelo para reconhecer todos os 10 dígitos com os parâmetros padrão (15 épocas, 2 camadas de convolução, etc.).

python run_experiment.py

2. Executar a Tarefa Binária:
Este comando treinará um modelo para distinguir apenas entre os dígitos '4' e '9', por 10 épocas.

python run_experiment.py --task binary --binary_digits 4,9 --epochs 10

3. Executar um Experimento Personalizado:
Treina um modelo multiclasse com apenas 1 camada de convolução, taxa de dropout de 0.2 e por 5 épocas.

python run_experiment.py --task multiclass --conv_layers 1 --dropout_rate 0.2 --epochs 5

Argumentos Disponíveis

Argumento

Tipo

Padrão

Descrição

--task

str

multiclass

Tipo de tarefa: multiclass ou binary.

--binary_digits

str

'4,9'

Dígitos para a tarefa binária (ex: "4,9").

--output_dir

str

results

Diretório raiz para salvar os resultados.

--epochs

int

15

Número de épocas de treinamento.

--batch_size

int

128

Tamanho do lote de treinamento.

--learning_rate

float

0.001

Taxa de aprendizado do otimizador Adam.

--conv_layers

int

2

Número de camadas de convolução (1 ou 2).

--dense_units

int

128

Número de neurônios na camada densa oculta.

--dropout_rate

float

0.4

Taxa de dropout para regularização.

Estrutura de Saída do Projeto
Para cada execução, uma nova pasta é criada dentro de results/, nomeada com um timestamp para evitar sobreposição de dados. Ex: results/20250616-194500_multiclass/.

Dentro desta pasta, você encontrará os seguintes artefatos:

hiperparametros.json: Arquivo JSON com a configuração exata de todos os parâmetros usados no experimento.

model_summary.txt: A arquitetura detalhada do modelo, gerada pelo model.summary().

pesos_iniciais.weights.h5: Os pesos do modelo antes do treinamento (inicialização aleatória).

pesos_finais.weights.h5: Os pesos do modelo após a conclusão do treinamento.

log_treinamento.csv: Um arquivo CSV com a perda (loss) e a acurácia de cada época, para treino e validação.

previsoes_teste.csv: As previsões do modelo para cada amostra do conjunto de teste.

relatorio_classificacao.txt: Relatório textual com métricas detalhadas (precisão, recall, F1-score) para cada classe.

tempo_treinamento.txt: O tempo total gasto no treinamento, em segundos.

curvas_aprendizado.png: Gráfico com as curvas de perda e acurácia.

matriz_confusao.png: Imagem da matriz de confusão.

curva_roc.png: (Apenas para tarefa binária) Gráfico da curva ROC com o valor de AUC.
