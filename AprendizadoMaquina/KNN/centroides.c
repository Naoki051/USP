#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_PONTOS 1000
#define NUM_CATEGORIAS 3

/* ============================
   ENUMS E ESTRUTURAS
   ============================ */

typedef enum {
    CENTRAL = 0,
    SUBURBANO = 1,
    INDUSTRIAL = 2
} Categoria;

typedef struct {
    double x;          // Distância do centro
    double y;          // Área do imóvel
    Categoria categoria;
} Ponto;

/* ============================
   PROTÓTIPOS
   ============================ */
int carregar_dataset(const char *nome_arquivo, Ponto dataset[], int max);
double distancia_euclidiana(Ponto a, Ponto b);
void treinar_centroides(Ponto dataset[], int total_treino, Ponto centroides[]);
int classificar_por_centroide(Ponto centroides[], Ponto teste);
void gerar_relatorio(FILE *f, int matriz[NUM_CATEGORIAS][NUM_CATEGORIAS], double tempo_ms);

/* ============================
   IMPLEMENTAÇÕES
   ============================ */

int carregar_dataset(const char *nome_arquivo, Ponto dataset[], int max) {
    FILE *file = fopen(nome_arquivo, "r");
    if (!file) return -1;

    char buffer[256];
    fgets(buffer, sizeof(buffer), file); 

    int i = 0;
    while (i < max && fscanf(file, "%lf,%lf,%d", &dataset[i].x, &dataset[i].y, (int *)&dataset[i].categoria) == 3) {
        i++;
    }
    fclose(file);
    return i;
}

double distancia_euclidiana(Ponto a, Ponto b) {
    double dx = a.x - b.x;
    double dy = a.y - b.y;
    return sqrt(dx * dx + dy * dy);
}

void treinar_centroides(Ponto dataset[], int total_treino, Ponto centroides[]) {
    int contagem[NUM_CATEGORIAS] = {0};

    for (int i = 0; i < NUM_CATEGORIAS; i++) {
        centroides[i].x = 0.0;
        centroides[i].y = 0.0;
        centroides[i].categoria = (Categoria)i;
    }

    for (int i = 0; i < total_treino; i++) {
        Categoria c = dataset[i].categoria;
        centroides[c].x += dataset[i].x;
        centroides[c].y += dataset[i].y;
        contagem[c]++;
    }

    for (int i = 0; i < NUM_CATEGORIAS; i++) {
        if (contagem[i] > 0) {
            centroides[i].x /= contagem[i];
            centroides[i].y /= contagem[i];
        }
    }
}

int classificar_por_centroid(Ponto centroides[], Ponto teste) {
    int melhor_cat = 0;
    double menor_dist = distancia_euclidiana(teste, centroides[0]);

    for (int i = 1; i < NUM_CATEGORIAS; i++) {
        double d = distancia_euclidiana(teste, centroides[i]);
        if (d < menor_dist) {
            menor_dist = d;
            melhor_cat = i;
        }
    }
    return melhor_cat;
}

void gerar_relatorio(FILE *f, int matriz[NUM_CATEGORIAS][NUM_CATEGORIAS], double tempo_ms) {
    int total = 0, acertos = 0;
    fprintf(f, "\n--- RELATORIO DE DESEMPENHO: CENTROIDES ---\n");
    fprintf(f, "Tempo de Treino + Classificacao: %.4f ms\n", tempo_ms);
    fprintf(f, "\nMatriz de Confusao:\n");
    fprintf(f, "R\\P |  0  |  1  |  2  \n");
    fprintf(f, "---------------------\n");

    for(int i=0; i<NUM_CATEGORIAS; i++) {
        fprintf(f, " %d  |", i);
        for(int j=0; j<NUM_CATEGORIAS; j++) {
            fprintf(f, " %3d |", matriz[i][j]);
            total += matriz[i][j];
            if(i == j) acertos += matriz[i][j];
        }
        fprintf(f, "\n");
    }

    double acuracia = (total > 0) ? (double)acertos / total : 0;
    fprintf(f, "\nAcuracia Global: %.2f%%\n\n", acuracia * 100);

    for(int i=0; i<NUM_CATEGORIAS; i++) {
        int vp = matriz[i][i];
        int soma_linha = 0, soma_coluna = 0;
        for(int j=0; j<NUM_CATEGORIAS; j++) {
            soma_linha += matriz[i][j];
            soma_coluna += matriz[j][i];
        }
        double prec = (soma_coluna > 0) ? (double)vp / soma_coluna : 0;
        double rec = (soma_linha > 0) ? (double)vp / soma_linha : 0;
        double f1 = (prec + rec > 0) ? 2 * (prec * rec) / (prec + rec) : 0;

        fprintf(f, "Classe %d -> Precisao: %.2f | Recall: %.2f | F1-Score: %.2f\n", i, prec, rec, f1);
    }
}

/* ============================
   FUNÇÃO PRINCIPAL
   ============================ */

int main(void) {
    Ponto dataset[MAX_PONTOS];
    Ponto centroides[NUM_CATEGORIAS];
    int matriz_confusao[NUM_CATEGORIAS][NUM_CATEGORIAS] = {0};

    int total = carregar_dataset("dataset.csv", dataset, MAX_PONTOS);
    if (total <= 0) {
        printf("Erro ao carregar dataset.\n");
        return 1;
    }

    int n_treino = (int)(total * 0.8);
    int n_teste = total - n_treino;

    clock_t start = clock();

    // 1. Treinar (calcular médias das classes usando 80% dos dados)
    treinar_centroides(dataset, n_treino, centroides);

    // 2. Testar (classificar os 20% restantes)
    for (int i = n_treino; i < total; i++) {
        int previsto = classificar_por_centroid(centroides, dataset[i]);
        int real = dataset[i].categoria;
        matriz_confusao[real][previsto]++;
    }

    clock_t end = clock();
    double tempo_ms = ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;

    // Saídas
    gerar_relatorio(stdout, matriz_confusao, tempo_ms);

    FILE *f = fopen("resultados_centroides.txt", "w");
    if (f) {
        gerar_relatorio(f, matriz_confusao, tempo_ms);
        fclose(f);
        printf("\nResultados salvos em 'resultados_centroides.txt'\n");
    }

    return 0;
}