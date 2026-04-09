#include <stdio.h>
#include <stdlib.h>

typedef struct{
    int numero;
    int duracao;
} Atualizacao;

typedef struct{
    Atualizacao *dados; // Vetor de estruturas
    int tamanho;
    int capacidade;
} Heap;

// 1. Alocação da Heap
Heap *criarHeap(int capacidade){
    Heap *h = (Heap *)malloc(sizeof(Heap));
    h->dados = (Atualizacao *)malloc(capacidade * sizeof(Atualizacao));
    h->tamanho = 0;
    h->capacidade = capacidade;
    return h;
}

void trocar(Atualizacao *a, Atualizacao *b){
    Atualizacao temp = *a;
    *a = *b;
    *b = temp;
}

// 2. Inserção (Mantendo a lógica de Max-Heap se quiser priorizar maior duração)
// Se quiser Min-Heap, basta inverter o '>' para '<'
void inserirAttHeap(Heap *h, Atualizacao nova){
    if (h->tamanho == h->capacidade)
        return;

    int i = h->tamanho;
    h->dados[i] = nova;
    h->tamanho++;

    // Sobe-Heap (Max-Heap)
    while (i != 0 && h->dados[i].duracao > h->dados[(i - 1) / 2].duracao){
        trocar(&h->dados[i], &h->dados[(i - 1) / 2]);
        i = (i - 1) / 2;
    }
}

// 3. Reordenamento para baixo (Max-Heap)
void descerNoHeap(Heap *h, int i){
    int maior = i;
    int esquerda = 2 * i + 1;
    int direita = 2 * i + 2;

    if (esquerda < h->tamanho && h->dados[esquerda].duracao > h->dados[maior].duracao)
        maior = esquerda;

    if (direita < h->tamanho && h->dados[direita].duracao > h->dados[maior].duracao)
        maior = direita;

    if (maior != i){
        trocar(&h->dados[i], &h->dados[maior]);
        descerNoHeap(h, maior);
    }
}

// 4. Remoção do topo
int executarAtualizacao(Heap *h){
    if (h->tamanho <= 0)
        return -1;

    int tempo = h->dados[0].duracao;
    h->dados[0] = h->dados[h->tamanho - 1];
    h->tamanho--;

    if (h->tamanho > 0){
        descerNoHeap(h, 0);
    }
    return tempo;
}

// 5. Caso Estático
// Supõe-se que receba um array de durações e processe-as
int *casoEstatico(Heap h, int horasDisponiveis){
    
}

void executarComando(Heap *h, char comando, int tempo, int numero, int novaDuracao){
    // comando i=inclisao, c=alteracao de duracar
    if (comando == 'i'){
        Atualizacao nova;
        nova.numero = numero;
        nova.duracao = novaDuracao;
        inserirAttHeap(h, nova);
    }else if (comando == 'c'){
        for (int i = 0; i < h->tamanho; i++){
            if (h->dados[i].numero == numero){
                h->dados[i].duracao = novaDuracao;
                // Tenta subir o nó
                int atual = i;
                while (atual != 0 && h->dados[atual].duracao > h->dados[(atual - 1) / 2].duracao){
                    trocar(&h->dados[atual], &h->dados[(atual - 1) / 2]);
                    atual = (atual - 1) / 2;
                }
                // Tenta descer o nó
                descerNoHeap(h, atual);
                break; // Encontrou, pode parar o loop
            }
        }
    }
}
