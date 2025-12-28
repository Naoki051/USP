#include <stdio.h>
#include <stdlib.h>
#include <time.h>

/**
 * Gera um dataset sintético para classificação de zonas urbanas.
 * @param nome Nome do arquivo .csv a ser criado.
 * @param n Número total de amostras.
 */
void gerar_csv_grande(const char *nome, int n) {
    FILE *f = fopen(nome, "w");
    if (f == NULL) {
        printf("Erro ao criar o arquivo!\n");
        return;
    }

    // Cabeçalho do CSV
    fprintf(f, "distancia,area,categoria\n");
    
    // Semente para números aleatórios baseada no tempo atual
    srand((unsigned int)time(NULL));

    for (int i = 0; i < n; i++) {
        int cat = i % 3; // Garante uma distribuição equilibrada entre as 3 classes
        double dist, area;
        
        if (cat == 0) { // CENTRAL: Perto e pequeno
            dist = (double)(rand() % 50) / 10.0;     // 0.0 a 5.0 km
            area = (double)(rand() % 41 + 30);       // 30 a 70 m2
        } 
        else if (cat == 1) { // SUBURBANO: Médio e espaçoso
            dist = (double)(rand() % 101 + 50) / 10.0; // 5.0 a 15.0 km
            area = (double)(rand() % 151 + 100);      // 100 a 250 m2
        } 
        else { // INDUSTRIAL: Longe e enorme
            dist = (double)(rand() % 301 + 200) / 10.0; // 20.0 a 50.0 km
            area = (double)(rand() % 2001 + 500);      // 500 a 2500 m2
        }

        fprintf(f, "%.2f,%.2f,%d\n", dist, area, cat);
    }

    fclose(f);
    printf("Arquivo '%s' gerado com %d amostras com sucesso!\n", nome, n);
}

int main(void) {
    int quantidade = 1000; // Agora você tem dados de sobra para treinar e testar!
    
    printf("Gerando dataset...\n");
    gerar_csv_grande("dataset.csv", quantidade);
    
    return 0;
}