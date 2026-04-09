#include <stdio.h>
#include <stdlib.h>

void explorarGrid(int** matrix, int rowStart, int colStart, int dim, int ordem, int** resposta) {
    // Caso Base: Menor unidade (Ordem 0 ou 1, dependendo do seu índice)
    if (dim == 1) {
        int valor = matrix[rowStart][colStart];
        resposta[ordem][valor]++;
        return;
    }
    int novaDim = dim / 2;
    int antes0 = resposta[ordem][0]; 
    int antes1 = resposta[ordem][1];
    // Busca o estado dos 4 quadrantes
    explorarGrid(matrix, rowStart, colStart, novaDim, ordem, resposta);
    explorarGrid(matrix, rowStart, colStart + novaDim, novaDim, ordem, resposta);
    explorarGrid(matrix, rowStart + novaDim, colStart, novaDim, ordem, resposta);
    explorarGrid(matrix, rowStart + novaDim, colStart + novaDim, novaDim, ordem, resposta);
    if (resposta[ordem][0] == antes0 + 4) {
        resposta[ordem][0] -= 4;     // Remove do nível atual
        resposta[ordem + 1][0] += 1; // "Promove" para o nível superior
    } else if (resposta[ordem][1] == antes1 + 4) {
        resposta[ordem][1] -= 4;
        resposta[ordem + 1][1] += 1;
    }    
}

int main() {
    int N = 4; // Tamanho da matriz (deve ser potência de 2)
    int niveis = 3; // Nível 0 (1x1), Nível 1 (2x2), Nível 2 (4x4)

    // 1. Alocação da Matriz de Entrada (4x4)
    // Exemplo: Um bloco 2x2 de '1's e o resto '0's
    int** matrix = (int**)malloc(N * sizeof(int*));
    for (int i = 0; i < N; i++) matrix[i] = (int*)malloc(N * sizeof(int));

    int dados[4][4] = {
        {1, 1, 0, 0},
        {1, 1, 0, 0},
        {0, 0, 0, 0},
        {0, 0, 0, 0}
    };

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            matrix[i][j] = dados[i][j];

    // 2. Alocação da Matriz Resposta [Nivel][Valor]
    // Linhas = níveis de profundidade, Colunas = valores possíveis (0 ou 1)
    int** resposta = (int**)calloc(niveis, sizeof(int*));
    for (int i = 0; i < niveis; i++) resposta[i] = (int*)calloc(2, sizeof(int));

    // 3. Execução
    printf("Explorando Grid %dx%d...\n", N, N);
    explorarGrid(matrix, 0, 0, N, 0, resposta);

    // 4. Exibição dos Resultados (Ajustado)
    for (int i = niveis - 1; i >= 0; i--) {
        int tamanho = 1 << i; // Calcula 2 elevado a i (ex: 1, 2, 4, 8...)
        printf("%dx%d %d %d\n", tamanho, tamanho, resposta[i][0], resposta[i][1]);
    }

    // Limpeza de memória
    for (int i = 0; i < N; i++) free(matrix[i]);
    free(matrix);
    for (int i = 0; i < niveis; i++) free(resposta[i]);
    free(resposta);

    return 0;
}
