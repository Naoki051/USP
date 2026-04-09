/* * ============================================================================
 * SISTEMA DE GERENCIAMENTO DE CAMPEONATO
 * ============================================================================
 * * REGRAS DE PONTUAÇÃO E ESTATÍSTICAS:
 * - Pontuação: Vitória (3 pts), Empate (1 pt), Derrota (0 pts).
 * - Saldo de Gols (SG): Gols Marcados (GM) - Gols Sofridos (GS).
 * - Média de Gols (Avg): GM / GS (calculada apenas se GS > 0, caso contrário -1).
 * * CRITÉRIOS DE ORDENAÇÃO (Min-Heap / Heapsort):
 * 1º Maior número de Pontos.
 * 2º Maior Saldo de Gols (Desempate).
 * * FUNCIONALIDADES DO MENU:
 * 1. Registrar Resultado: Entrada (Time1, Time2, Gols1, Gols2).
 * 2. Imprimir Tabela: Exibe classificação completa ordenada.
 * 3. Consultar Time: Exibe estatísticas detalhadas de um clube específico.
 * 4. Salvar Dados: Exporta a tabela atual para um arquivo .txt.
 * * ESTRUTURA DE DADOS: 
 * - Alocação dinâmica com Array de Ponteiros para eficiência em trocas e inserções.
 * ============================================================================
 */
#include <stdbool.h> 
#include <stdio.h>   
#include <stdlib.h>  
#include <string.h>  

// Nome padrão para exportação de dados
const char* OUTPUT_FILE = "resultadoJogos.txt";

typedef struct time {
    char* nomeTime;
    int golsMarcados;
    int golsSofridos;
    int saldoGols;
    int vitorias;
    int empates;
    float avrageGols; // Razão entre gols marcados e sofridos
    int pontos;
} Time;

typedef struct tabela {
    int qtdTimes;    // Contador de times registrados
    Time** times;    // Array de ponteiros: armazena endereços das structs 'Time'
} Tabela;

void inicializaTime(Time* t, char* nomeTime) {
    t->nomeTime = strdup(nomeTime); // Duplica a string para evitar erros de escopo
    t->golsMarcados = 0;
    t->golsSofridos = 0;
    t->saldoGols = 0;
    t->avrageGols = -1;
    t->vitorias = 0;
    t->empates = 0;
    t->pontos = 0;
}

void adicionarNovoTime(Tabela* resultado, Time* novoTime) {
    // Expande o array de endereços usando um ponteiro temporário
    Time** temp = (Time**)realloc(resultado->times, sizeof(Time*) * (resultado->qtdTimes + 1));
    
    if (temp != NULL) {
        // Recebe o novo ponteiro para o array expandido
        resultado->times = temp;
        resultado->times[resultado->qtdTimes] = novoTime;
        resultado->qtdTimes++;
    } else {
        // Tratamento de erro
        printf("Erro: Falha critica de memoria ao adicionar time.\n");
    }
}

Time* buscarTime(Tabela* resultado, char* nomeTime) {
    // Percorre a tabela para verificar se o time já existe (busca linear)
    for (int i = 0; i < resultado->qtdTimes; i++) {
        // Compara o conteúdo das strings; retorna o ponteiro se encontrar o time
        if (strcmp(resultado->times[i]->nomeTime, nomeTime) == 0) {
            return resultado->times[i];
        }
    }
    return NULL;
}

void registrarPartida(Tabela* resultado, char* nomeTime1, char* nomeTime2, int gols1, int gols2) {
    // 1. Tenta buscar os times
    Time* t1 = buscarTime(resultado, nomeTime1);
    Time* t2 = buscarTime(resultado, nomeTime2);
    // 2. Se o Time 1 não existe, cria e adiciona
    if (t1 == NULL) {
        t1 = (Time*)malloc(sizeof(Time)); 
        if (t1) {
            inicializaTime(t1, nomeTime1);
            adicionarNovoTime(resultado, t1); // PASSAR O OBJETO t1, NÃO O NOME
        }
    }
    // 3. Se o Time 2 não existe, cria e adiciona
    if (t2 == NULL) {
        t2 = (Time*)malloc(sizeof(Time)); 
        if (t2) {
            inicializaTime(t2, nomeTime2);
            adicionarNovoTime(resultado, t2); // PASSAR O OBJETO t2, NÃO O NOME
        }
    }
    // Validação de segurança para evitar operações em ponteiros nulos
    if (!t1 || !t2) return; 
    // Atualiza estatísticas de desempenho (Vitórias/Empates) conforme o placar
    if (gols1 > gols2) t1->vitorias++;
    else if (gols1 < gols2) t2->vitorias++;
    else {
        t1->empates++;
        t2->empates++;
    }
    // Acumula gols e recalcula o saldo de gols (GM - GS)
    t1->golsMarcados += gols1;
    t2->golsMarcados += gols2;
    t1->golsSofridos += gols2;
    t2->golsSofridos += gols1;
    t1->saldoGols = t1->golsMarcados - t1->golsSofridos;
    t2->saldoGols = t2->golsMarcados - t2->golsSofridos;
    // Calcula a média de gols (Average) usando cast para float para garantir precisão decimal
    if (t1->golsSofridos > 0) 
        t1->avrageGols = (float)t1->golsMarcados / t1->golsSofridos;
    if (t2->golsSofridos > 0) 
        t2->avrageGols = (float)t2->golsMarcados / t2->golsSofridos;
    // Atualiza a pontuação total baseada nas regras: Vitória = 3pts, Empate = 1pt
    t1->pontos = (t1->vitorias * 3) + t1->empates;
    t2->pontos = (t2->vitorias * 3) + t2->empates;
}

// Troca apenas os endereços armazenados no array, evitando a cópia pesada das structs
void trocarPonteiros(Time** a, Time** b) {
    Time* temp = *a;
    *a = *b;
    *b = temp;
}

// Mantém a propriedade do Min-Heap: o time com menor desempenho sobe para a raiz
void minHeapify(Tabela* tab, int n, int i) {
    int menor = i;          // Assume o nó atual como o menor
    int esq = 2 * i + 1;    // Cálculo do índice do filho à esquerda
    int dir = 2 * i + 2;    // Cálculo do índice do filho à direita
    // Critério de desempate: Pontos > Saldo > Vitórias
    if (tab->times[esq]->pontos < tab->times[menor]->pontos) {
        menor = esq;
    } else if (tab->times[esq]->pontos == tab->times[menor]->pontos) {
        if (tab->times[esq]->saldoGols < tab->times[menor]->saldoGols) {
            menor = esq;
        } else if (tab->times[esq]->saldoGols == tab->times[menor]->saldoGols) {
            if (tab->times[esq]->vitorias < tab->times[menor]->vitorias) {
                menor = esq;
            }
        }
    }
    // Verifica se o filho direito é o menor entre pai, esquerda e direita
    if (dir < n) {
        if (tab->times[dir]->pontos < tab->times[menor]->pontos) {
            menor = dir;
        }
        else if (tab->times[dir]->pontos == tab->times[menor]->pontos) {
            if (tab->times[dir]->saldoGols < tab->times[menor]->saldoGols) {
                menor = dir;
            }
        }
    }
    // Se o menor não for o pai, realiza a troca e propaga a alteração para os níveis inferiores
    if (menor != i) {
        trocarPonteiros(&tab->times[i], &tab->times[menor]);
        minHeapify(tab, n, menor); // Chamada recursiva para garantir a integridade do Heap
    }
}

void ordenarTimes(Tabela* resultado) {
    int n = resultado->qtdTimes;
    // 1. Transforma o array em um Min-Heap: o time com menor desempenho vai para a raiz (índice 0)
    for (int i = n / 2 - 1; i >= 0; i--) {
        minHeapify(resultado, n, i);
    }
    // 2. Processo de extração: move o menor elemento para o final do array sucessivamente
    for (int i = n - 1; i > 0; i--) {
        // O "pior" time (raiz) é trocado com o último elemento não ordenado
        trocarPonteiros(&resultado->times[0], &resultado->times[i]);
        // Restaura a propriedade de Min-Heap na parte restante do array (tamanho 'i')
        minHeapify(resultado, i, 0);
    }
    // RESULTADO: O array termina ordenado de forma decrescente (do maior para o menor), 
    // colocando o líder na primeira posição e o lanterna na última.
}

void imprimirTabela(Tabela* resultado) {
    // Exibe o cabeçalho com alinhamento à esquerda (%-) e larguras fixas para organizar as colunas
    printf("\n%-4s %-15s %-3s %-3s %-3s %-3s %-6s %-3s %-3s\n", 
           "Pos", "Time", "Pts", "SG", "GM", "GS", "Avg", "V", "E");
    printf("------------------------------------------------------------\n");

    // Itera sobre o array de ponteiros até o total de times registrados
    for (int i = 0; i < resultado->qtdTimes; i++) {
        // Acessa o endereço do time armazenado no índice atual do array de ponteiros
        Time* t = resultado->times[i]; 
        
        // Imprime os dados acessando os campos da struct via ponteiro (operador ->)
        // %-15s garante que o nome do time ocupe 15 espaços, mantendo a tabela alinhada
        printf("%-4d %-15s %-3d %-3d %-3d %-3d %-6.2f %-3d %-3d\n", 
               i + 1, 
               t->nomeTime, 
               t->pontos, 
               t->saldoGols, 
               t->golsMarcados, 
               t->golsSofridos, 
               t->avrageGols, 
               t->vitorias, 
               t->empates);
    }
}

bool salvarTabela(Tabela* resultado, const char* nomeArquivo) {
    // Abre o arquivo no modo de escrita ("w"); se o arquivo existir, será sobrescrito
    FILE* arquivo = fopen(nomeArquivo, "w");
    
    // Verifica se o sistema operacional permitiu a criação/abertura do arquivo
    if (arquivo == NULL) {
        printf("Erro ao abrir o arquivo para salvar.\n");
        return false;
    }

    // Grava o cabeçalho formatado no arquivo para garantir a legibilidade do relatório
    fprintf(arquivo, "%-15s %-3s %-3s %-3s %-3s %-6s %-3s %-3s\n", 
            "Time", "Pts", "SG", "GM", "GS", "Avg", "V", "E");
    fprintf(arquivo, "------------------------------------------------------------\n");

    // Itera pelo array de ponteiros, extraindo os dados de cada struct 'Time'
    for (int i = 0; i < resultado->qtdTimes; i++) {
        Time* t = resultado->times[i];
        
        // fprintf direciona a saída formatada para o fluxo do arquivo em vez do console
        fprintf(arquivo, "%-15s %-3d %-3d %-3d %-3d %-6.2f %-3d %-3d\n", 
                t->nomeTime, 
                t->pontos, 
                t->saldoGols, 
                t->golsMarcados, 
                t->golsSofridos, 
                t->avrageGols, 
                t->vitorias, 
                t->empates);
    }

    // Fecha o fluxo do arquivo para garantir que todos os dados sejam gravados fisicamente no disco
    fclose(arquivo);
    return true; // Indica sucesso na operação de salvamento
}

void imprimirMenu(Tabela* tabela) {
    int opcao;
    char nome1[50], nome2[50]; // Buffers temporários para leitura de nomes
    int gols1, gols2;

    do {
        // Exibe as opções de interação para o usuário via console
        printf("\n========= GERENCIADOR DE CAMPEONATO =========\n");
        printf("1. Registrar Resultado da Partida\n");
        printf("2. Imprimir Tabela Completa (Ordenada)\n");
        printf("3. Buscar Dados de um Time\n");
        printf("4. Salvar Dados em Arquivo\n");
        printf("0. Sair\n");
        printf("Escolha uma opcao: ");
        scanf("%d", &opcao);
        setbuf(stdin, NULL); // Evita problemas de leitura ao limpar o caractere '\n' residual do buffer
        switch (opcao) {
            case 1:
                printf("Time da Casa: ");
                // O espaço antes do % limpa qualquer 'Enter' que ficou sobrando
                scanf(" %49[^\n]", nome1); 
                
                printf("Gols %s: ", nome1);
                // Verificamos se o usuário digitou um número válido
                if (scanf("%d", &gols1) != 1) {
                    printf("Erro: Digite apenas numeros para os gols!\n");
                    while (getchar() != '\n'); // Limpa o buffer se o usuário digitou letras
                    break; 
                }

                printf("Time Visitante: ");
                scanf(" %49[^\n]", nome2); 
                
                printf("Gols %s: ", nome2);
                if (scanf("%d", &gols2) != 1) {
                    printf("Erro: Digite apenas numeros para os gols!\n");
                    while (getchar() != '\n');
                    break;
                }

                registrarPartida(tabela, nome1, nome2, gols1, gols2);
                printf("\n>>> Partida registrada: %s %d x %d %s\n", nome1, gols1, gols2, nome2);
                break;
            case 2:
                // Aplica Heapsort antes da exibição para garantir classificação atualizada
                ordenarTimes(tabela); 
                imprimirTabela(tabela);
                break;
            case 3:
                printf("Digite o nome do time: ");
                scanf("%49s", nome1);
                Time* t = buscarTime(tabela, nome1);
                
                if (t != NULL) {
                    printf("\n%-15s %-3s %-3s %-3s %-3s %-6s\n", "Time", "Pts", "SG", "GM", "GS", "Avg");
                    printf("%-15s %-3d %-3d %-3d %-3d %-6.2f\n", 
                        t->nomeTime, t->pontos, t->saldoGols, t->golsMarcados, t->golsSofridos, t->avrageGols);
                } else {
                    printf("Time '%s' nao encontrado na tabela.\n", nome1);
                }
                break;
            case 4:
                // Exporta o estado atual da memória para o disco rígido
                if (salvarTabela(tabela, "campeonato.txt")) {
                    printf("Dados salvos em 'campeonato.txt'!\n");
                }
                break;
            case 0:
                printf("Saindo e limpando memoria...\n");
                break;

            default:
                printf("Opcao invalida!\n");
        }
    } while (opcao != 0); // Mantém o programa ativo até o comando de saída explícito
}

void limparTabela(Tabela* tab) {
    // Verifica se a tabela ou o array de ponteiros é nulo antes de prosseguir
    if (tab == NULL || tab->times == NULL) return;

    // Percorre cada posição do array de ponteiros
    for (int i = 0; i < tab->qtdTimes; i++) {
        // 1. Libera o nome do time (alocado dinamicamente pelo strdup)
        if (tab->times[i]->nomeTime != NULL) {
            free(tab->times[i]->nomeTime);
        }
        // 2. Libera a struct Time individualmente
        free(tab->times[i]);
    }

    // 3. Libera o array que armazenava os endereços (os ponteiros)
    free(tab->times);

    // 4. Reseta os valores da estrutura para garantir segurança (evita ponteiros pendentes)
    tab->times = NULL;
    tab->qtdTimes = 0;
}

int main() {
    Tabela minhaTabela = {0, NULL}; // Inicia com 0 times e ponteiro nulo
    
    imprimirMenu(&minhaTabela);
    
    limparTabela(&minhaTabela); // Função de free que discutimos antes
    return 0;
}
