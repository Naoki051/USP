#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define MAX_PONTOS 1000
#define NUM_CATEGORIAS 3

/* ============================
   ESTRUTURAS
   ============================ */
typedef enum {
    CENTRAL = 0,
    SUBURBANO = 1,
    INDUSTRIAL = 2
} Categoria;

typedef struct {
    double x;
    double y;
    Categoria categoria;
} Ponto;

typedef struct {
    double distancia;
    Categoria categoria;
} Vizinho;

/* ============================
   PROTÓTIPOS
   ============================ */
int carregar_dataset(const char *nome_arquivo, Ponto dataset[], int max);
double distancia_euclidiana(Ponto a, Ponto b);
void ordenar_top_k(Vizinho v[], int k);
int classificar_knn(Ponto dataset[], int total_treino, Ponto teste, int k);
void gerar_relatorio(FILE *f, int matriz[NUM_CATEGORIAS][NUM_CATEGORIAS], double tempo_ms, int k);

/* ============================
   IMPLEMENTAÇÕES
   ============================ */

int carregar_dataset(const char *nome_arquivo, Ponto dataset[], int max) {
    FILE *arquivo = fopen(nome_arquivo, "r");
    if (!arquivo) return -1;

    char buffer[1024];
    fgets(buffer, sizeof(buffer), arquivo); // pula cabeçalho

    int i = 0;
    while (i < max && fscanf(arquivo, "%lf,%lf,%d", &dataset[i].x, &dataset[i].y, (int *)&dataset[i].categoria) == 3) {
        i++;
    }
    fclose(arquivo);
    return i;
}

double distancia_euclidiana(Ponto a, Ponto b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

void ordenar_top_k(Vizinho v[], int k) {
    for (int i = 0; i < k - 1; i++) {
        for (int j = 0; j < k - i - 1; j++) {
            if (v[j].distancia > v[j + 1].distancia) {
                Vizinho temp = v[j];
                v[j] = v[j + 1];
                v[j + 1] = temp;
            }
        }
    }
}

int classificar_knn(Ponto dataset[], int total_treino, Ponto teste, int k) {
    Vizinho vizinhos[k];
    int preenchidos = 0;

    for (int i = 0; i < total_treino; i++) {
        double d = distancia_euclidiana(dataset[i], teste);
        if (preenchidos < k) {
            vizinhos[preenchidos++] = (Vizinho){d, dataset[i].categoria};
            if (preenchidos == k) ordenar_top_k(vizinhos, k);
        } else if (d < vizinhos[k - 1].distancia) {
            vizinhos[k - 1] = (Vizinho){d, dataset[i].categoria};
            ordenar_top_k(vizinhos, k);
        }
    }

    int votos[NUM_CATEGORIAS] = {0};
    for (int i = 0; i < k; i++) votos[vizinhos[i].categoria]++;

    int vencedor = 0;
    for (int i = 1; i < NUM_CATEGORIAS; i++) {
        if (votos[i] > votos[vencedor]) vencedor = i;
    }
    return vencedor;
}

// Função genérica que escreve em um stream (console ou arquivo)
void gerar_relatorio(FILE *f, int matriz[NUM_CATEGORIAS][NUM_CATEGORIAS], double tempo_ms, int k) {
    int total = 0, acertos = 0;
    fprintf(f, "\n--- RELATORIO DE DESEMPENHO KNN (K=%d) ---\n", k);
    fprintf(f, "Tempo de Execucao Total: %.4f ms\n", tempo_ms);
    fprintf(f, "\nMatriz de Confusao (Linha: Real | Coluna: Previsto):\n");
    fprintf(f, "R\\P |  0  |  1  |  2  \n");
    fprintf(f, "---------------------\n");

    for (int i = 0; i < NUM_CATEGORIAS; i++) {
        fprintf(f, " %d  |", i);
        for (int j = 0; j < NUM_CATEGORIAS; j++) {
            fprintf(f, " %3d |", matriz[i][j]);
            total += matriz[i][j];
            if (i == j) acertos += matriz[i][j];
        }
        fprintf(f, "\n");
    }

    double acuracia = (total > 0) ? (double)acertos / total : 0;
    fprintf(f, "\nAcuracia Global: %.2f%%\n\n", acuracia * 100);

    for (int i = 0; i < NUM_CATEGORIAS; i++) {
        int vp = matriz[i][i];
        int soma_linha = 0, soma_coluna = 0;
        for (int j = 0; j < NUM_CATEGORIAS; j++) {
            soma_linha += matriz[i][j];
            soma_coluna += matriz[j][i];
        }
        double prec = (soma_coluna > 0) ? (double)vp / soma_coluna : 0;
        double rec = (soma_linha > 0) ? (double)vp / soma_linha : 0;
        double f1 = (prec + rec > 0) ? 2 * (prec * rec) / (prec + rec) : 0;

        fprintf(f, "Classe %d -> Precisao: %.2f | Recall: %.2f | F1-Score: %.2f\n", i, prec, rec, f1);
    }
    fprintf(f, "------------------------------------------\n");
}

/* ============================
   FUNÇÃO PRINCIPAL
   ============================ */
int main(void) {
    Ponto dataset[MAX_PONTOS];
    int matriz_confusao[NUM_CATEGORIAS][NUM_CATEGORIAS] = {0};
    int k_vizinhos = 3;

    int total = carregar_dataset("dataset.csv", dataset, MAX_PONTOS);
    if (total <= 0) {
        printf("Erro ao carregar o dataset. Verifique se 'dataset.csv' existe.\n");
        return 1;
    }

    // Dividindo o dataset: 80% treino, 20% teste
    int n_treino = (int)(total * 0.8);
    int n_teste = total - n_treino;

    printf("Iniciando KNN: %d treino, %d teste...\n", n_treino, n_teste);

    clock_t start = clock(); // Início da medição de tempo

    // Classificando os pontos de teste (os últimos 20% do dataset)
    for (int i = n_treino; i < total; i++) {
        int previsto = classificar_knn(dataset, n_treino, dataset[i], k_vizinhos);
        int real = dataset[i].categoria;
        matriz_confusao[real][previsto]++;
    }

    clock_t end = clock(); // Fim da medição
    double tempo_total_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // 1. Mostrar no Console
    gerar_relatorio(stdout, matriz_confusao, tempo_total_ms, k_vizinhos);

    // 2. Salvar em Arquivo TXT
    FILE *txt = fopen("resultados_knn.txt", "w");
    if (txt) {
        gerar_relatorio(txt, matriz_confusao, tempo_total_ms, k_vizinhos);
        fclose(txt);
        printf("\nResultados salvos com sucesso em 'resultados_knn.txt'\n");
    } else {
        printf("\nErro ao criar arquivo de resultados.\n");
    }

    return 0;
}