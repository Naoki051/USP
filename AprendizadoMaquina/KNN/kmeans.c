#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#define MAX_PONTOS 1000
#define K 3
#define MAX_ITER 100
#define NUM_CATEGORIAS 3

/* ============================
   ESTRUTURAS
   ============================ */

typedef struct {
    double x, y;           // Valores originais
    double x_norm, y_norm; // Valores normalizados (0-1)
    int categoria_real;    // Classe verdadeira do CSV
    int cluster;           // Grupo atribuído pelo K-means
} Ponto;

/* ============================
   PROTÓTIPOS
   ============================ */
int carregar_dataset(const char *nome_arquivo, Ponto dataset[], int max);
void normalizar_minmax(Ponto dataset[], int n);
double distancia_euclidiana(double x1, double y1, double x2, double y2);
void executar_kmeans(Ponto dataset[], int n);
void gerar_relatorio(FILE *f, int matriz[NUM_CATEGORIAS][NUM_CATEGORIAS], double tempo_ms);
void mapear_clusters(Ponto dataset[], int n, int mapa[K]);

/* ============================
   IMPLEMENTAÇÕES
   ============================ */

int carregar_dataset(const char *nome_arquivo, Ponto dataset[], int max) {
    FILE *arquivo = fopen(nome_arquivo, "r");
    if (!arquivo) return -1;
    char cabecalho[128];
    fgets(cabecalho, sizeof(cabecalho), arquivo);
    int i = 0;
    while (i < max && fscanf(arquivo, "%lf,%lf,%d", &dataset[i].x, &dataset[i].y, &dataset[i].categoria_real) == 3) {
        dataset[i].cluster = -1;
        i++;
    }
    fclose(arquivo);
    return i;
}

void normalizar_minmax(Ponto dataset[], int n) {
    if (n <= 0) return;
    double min_x = dataset[0].x, max_x = dataset[0].x;
    double min_y = dataset[0].y, max_y = dataset[0].y;
    for (int i = 1; i < n; i++) {
        if (dataset[i].x < min_x) min_x = dataset[i].x;
        if (dataset[i].x > max_x) max_x = dataset[i].x;
        if (dataset[i].y < min_y) min_y = dataset[i].y;
        if (dataset[i].y > max_y) max_y = dataset[i].y;
    }
    double range_x = max_x - min_x;
    double range_y = max_y - min_y;
    for (int i = 0; i < n; i++) {
        dataset[i].x_norm = (range_x == 0) ? 0.0 : (dataset[i].x - min_x) / range_x;
        dataset[i].y_norm = (range_y == 0) ? 0.0 : (dataset[i].y - min_y) / range_y;
    }
}

double distancia_euclidiana(double x1, double y1, double x2, double y2) {
    return sqrt(pow(x1 - x2, 2) + pow(y1 - y2, 2));
}

void executar_kmeans(Ponto dataset[], int n) {
    double cent_x[K], cent_y[K];
    for (int i = 0; i < K; i++) {
        cent_x[i] = dataset[i].x_norm;
        cent_y[i] = dataset[i].y_norm;
    }
    for (int iter = 0; iter < MAX_ITER; iter++) {
        bool houve_mudanca = false;
        for (int i = 0; i < n; i++) {
            int melhor_cluster = 0;
            double menor_dist = distancia_euclidiana(dataset[i].x_norm, dataset[i].y_norm, cent_x[0], cent_y[0]);
            for (int j = 1; j < K; j++) {
                double d = distancia_euclidiana(dataset[i].x_norm, dataset[i].y_norm, cent_x[j], cent_y[j]);
                if (d < menor_dist) {
                    menor_dist = d;
                    melhor_cluster = j;
                }
            }
            if (dataset[i].cluster != melhor_cluster) {
                dataset[i].cluster = melhor_cluster;
                houve_mudanca = true;
            }
        }
        if (!houve_mudanca) break;
        double soma_x[K] = {0}, soma_y[K] = {0};
        int contagem[K] = {0};
        for (int i = 0; i < n; i++) {
            int c = dataset[i].cluster;
            soma_x[c] += dataset[i].x_norm;
            soma_y[c] += dataset[i].y_norm;
            contagem[c]++;
        }
        for (int j = 0; j < K; j++) {
            if (contagem[j] > 0) {
                cent_x[j] = soma_x[j] / contagem[j];
                cent_y[j] = soma_y[j] / contagem[j];
            }
        }
    }
}

// Vincula cada cluster encontrado à categoria real mais frequente nele
void mapear_clusters(Ponto dataset[], int n, int mapa[K]) {
    for (int c = 0; c < K; c++) {
        int contagem[NUM_CATEGORIAS] = {0};
        for (int i = 0; i < n; i++) {
            if (dataset[i].cluster == c) {
                contagem[dataset[i].categoria_real]++;
            }
        }
        int melhor_cat = 0;
        for (int cat = 1; cat < NUM_CATEGORIAS; cat++) {
            if (contagem[cat] > contagem[melhor_cat]) melhor_cat = cat;
        }
        mapa[c] = melhor_cat;
    }
}

void gerar_relatorio(FILE *f, int matriz[NUM_CATEGORIAS][NUM_CATEGORIAS], double tempo_ms) {
    int total = 0, acertos = 0;
    fprintf(f, "\n--- RELATORIO DE DESEMPENHO: K-MEANS NAO SUPERVISIONADO ---\n");
    fprintf(f, "Tempo Total de Processamento: %.4f ms\n", tempo_ms);
    fprintf(f, "\nMatriz de Confusao (Mapeada):\n");
    fprintf(f, "R\\P |  0  |  1  |  2  \n---------------------\n");
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
        int vp = matriz[i][i], sl = 0, sc = 0;
        for (int j = 0; j < NUM_CATEGORIAS; j++) { sl += matriz[i][j]; sc += matriz[j][i]; }
        double p = (sc > 0) ? (double)vp / sc : 0, r = (sl > 0) ? (double)vp / sl : 0;
        double f1 = (p + r > 0) ? 2 * (p * r) / (p + r) : 0;
        fprintf(f, "Classe %d -> Precisao: %.2f | Recall: %.2f | F1-Score: %.2f\n", i, p, r, f1);
    }
}

/* ============================
   FUNÇÃO PRINCIPAL
   ============================ */

int main(void) {
    Ponto dataset[MAX_PONTOS];
    int matriz_confusao[NUM_CATEGORIAS][NUM_CATEGORIAS] = {0};
    int mapa_cluster_para_cat[K];

    int n = carregar_dataset("dataset.csv", dataset, MAX_PONTOS);
    if (n <= 0) return 1;

    clock_t start = clock();
    
    normalizar_minmax(dataset, n);
    executar_kmeans(dataset, n);
    mapear_clusters(dataset, n, mapa_cluster_para_cat);

    // Preenche matriz usando o mapa para converter Cluster -> Categoria
    for (int i = 0; i < n; i++) {
        int previsto = mapa_cluster_para_cat[dataset[i].cluster];
        int real = dataset[i].categoria_real;
        matriz_confusao[real][previsto]++;
    }

    clock_t end = clock();
    double tempo_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    gerar_relatorio(stdout, matriz_confusao, tempo_ms);

    FILE *f = fopen("resultados_kmeans.txt", "w");
    if (f) { gerar_relatorio(f, matriz_confusao, tempo_ms); fclose(f); }

    return 0;
}