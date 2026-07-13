#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/* Metadados do problema, difundidos do rank 0 para os demais. */
typedef struct {
    int n;  /* número de pontos            */
    int d;  /* número de dimensões         */
    int k;  /* número de clusters (K)      */
} Meta;

/* Aborta toda a execução MPI com uma mensagem de erro formatada.
 * Usado em condições irrecuperáveis (arquivo inválido, argumentos ruins). */
#define DIE(...) do {                                  \
        fprintf(stderr, "[erro] " __VA_ARGS__);        \
        fprintf(stderr, "\n");                         \
        MPI_Abort(MPI_COMM_WORLD, 1);                  \
    } while (0)

#endif /* COMMON_H */
