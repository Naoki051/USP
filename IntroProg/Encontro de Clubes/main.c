/*
 * PROJETO: Sincronização de Clubes Escolares
 * -----------------------------------------
 * OBJETIVO: 
 * Calcular e exibir todas as datas de encontro simultâneo de cinco clubes 
 * escolares dentro do mesmo ano civil da data informada (DD/MM/AAAA).
 *
 * REGRAS DE FREQUÊNCIA (Ciclos de Reunião):
 * - Esportes:   Intervalos de 2 dias (Dia sim, dia não)
 * - Literatura: Intervalos de 3 dias
 * - Fotografia: Intervalos de 4 dias
 * - Xadrez:     Intervalos de 5 dias
 * - Música:     Intervalos de 6 dias
 *
 * LÓGICA MATEMÁTICA:
 * O encontro ocorre em intervalos baseados no Mínimo Múltiplo Comum (MMC).
 * MMC(2, 3, 4, 5, 6) = 60 dias.
 */

#include <stdio.h>
#include <stdbool.h>

// --- Estrutura de Dados ---

typedef struct {
    int dia, mes, ano;
} Data;

// --- Funções Auxiliares de Calendário ---

// Verifica se o ano é bissexto (Regra: divisível por 4 e não por 100, ou divisível por 400)
bool ehBissexto(int ano) {
    return (ano % 4 == 0 && ano % 100 != 0) || (ano % 400 == 0);
}

// Retorna a quantidade de dias de um mês específico
int diasNoMes(int mes, int ano) {
    int meses[] = {31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31};
    if (mes == 2 && ehBissexto(ano)) {
        return 29;
    }
    return meses[mes - 1];
}

// Valida se a data inserida pelo usuário existe no calendário
bool validarData(Data d) {
    if (d.ano < 1 || d.mes < 1 || d.mes > 12) return false;
    int maxDia = diasNoMes(d.mes, d.ano);
    return (d.dia >= 1 && d.dia <= maxDia);
}

// --- Lógica Principal de Datas ---

// Avança a data em N dias, respeitando viradas de mês e ano
Data adicionarDias(Data d, int n) {
    int mesAtual = d.mes;
    int anoAtual = d.ano;
    
    // 1. Calcula quantos dias faltam para o fim do mês atual
    int diasRestantesMes = diasNoMes(mesAtual, anoAtual) - d.dia;
    
    // 2. Se o salto cabe no mês atual
    if (n <= diasRestantesMes) {
        d.dia += n;
        return d;
    }
    
    // 3. Caso contrário, "gastamos" os dias para pular para o dia 1 do mês seguinte
    int saldoDias = n - (diasRestantesMes + 1);
    mesAtual++;
    if (mesAtual > 12) { mesAtual = 1; anoAtual++; }

    // 4. Subtrai meses inteiros enquanto o saldo for maior que o mês atual
    while (mesAtual <= 12 && saldoDias >= diasNoMes(mesAtual, anoAtual)) {
        saldoDias -= diasNoMes(mesAtual, anoAtual);
        mesAtual++;
    }

    // 5. Verificação de estouro: se o cálculo ultrapassou o ano original
    if (mesAtual > 12 || anoAtual > d.ano) {
        Data erro = {0, 0, -1}; 
        return erro;
    }

    Data final = {saldoDias + 1, mesAtual, anoAtual};
    return final;
}

// --- Interface do Usuário ---

int main(int argc, char const *argv[]) {
    Data dataInicial;

    // Entrada de dados com validação
    do {
        printf("Digite a data inicial (DD MM AAAA): ");
        if (scanf("%d %d %d", &dataInicial.dia, &dataInicial.mes, &dataInicial.ano) != 3) {
            printf("Formato invalido. Use numeros.\n");
            while(getchar() != '\n'); // Limpa o buffer de entrada
            continue;
        }

        if (!validarData(dataInicial)) {
            printf("ERRO: Data %02d/%02d/%04d nao existe. Tente novamente.\n", 
                    dataInicial.dia, dataInicial.mes, dataInicial.ano);
        }
    } while (!validarData(dataInicial));

    printf("\nDatas de encontro dos clubes em %04d:\n", dataInicial.ano);
    printf("--------------------------------------\n");

    Data proximo = dataInicial;

    // Loop de cálculo baseado no MMC = 60
    while (proximo.ano != -1) {
        printf("Encontro: %02d/%02d/%04d\n", proximo.dia, proximo.mes, proximo.ano);
        
        // Tenta calcular o próximo encontro (60 dias depois)
        proximo = adicionarDias(proximo, 60);
    }

    printf("--------------------------------------\n");
    printf("Fim do calendario para o ano de %d.\n", dataInicial.ano);

    return 0;
}