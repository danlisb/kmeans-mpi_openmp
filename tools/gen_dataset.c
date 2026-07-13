/* gen_dataset — gera um CSV sintético com K blobs gaussianos bem separados.
 *
 * Uso:
 *   ./gen_dataset <n_points> <n_dims> <k_clusters> <saida.csv> [seed=42] [spread=1.0]
 *
 * Saída: cabeçalho "feature_0,...,feature_{d-1}" seguido de n_points linhas.
 * Determinístico para um mesmo seed (usa srand/rand).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Amostra de uma normal(mean, sd) por Box-Muller. */
static double gaussian(double mean, double sd)
{
    double u1 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
    double u2 = (rand() + 1.0) / ((double)RAND_MAX + 2.0);
    return mean + sd * sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
}

int main(int argc, char **argv)
{
    if (argc < 5) {
        fprintf(stderr,
                "uso: %s <n_points> <n_dims> <k_clusters> <saida.csv> [seed=42] [spread=1.0]\n",
                argv[0]);
        return 1;
    }

    long   n      = atol(argv[1]);
    int    d      = atoi(argv[2]);
    int    k      = atoi(argv[3]);
    const char *out = argv[4];
    unsigned seed = (argc > 5) ? (unsigned)atoi(argv[5]) : 42u;
    double spread = (argc > 6) ? atof(argv[6]) : 1.0;

    if (n < 1 || d < 1 || k < 1) {
        fprintf(stderr, "[erro] n_points, n_dims e k_clusters devem ser >= 1\n");
        return 1;
    }

    srand(seed);

    /* Centros dos clusters: bem separados numa "diagonal" escalada.
     * Distância entre clusters vizinhos ~ 15*sqrt(d) >> spread. */
    double *centers = malloc((size_t)k * d * sizeof(double));
    if (!centers) { fprintf(stderr, "[erro] sem memória\n"); return 1; }
    for (int c = 0; c < k; c++)
        for (int j = 0; j < d; j++)
            centers[c * d + j] = (c + 1) * 15.0 + j * 2.0;

    FILE *f = fopen(out, "w");
    if (!f) { fprintf(stderr, "[erro] não foi possível abrir %s\n", out); free(centers); return 1; }

    /* Cabeçalho. */
    for (int j = 0; j < d; j++)
        fprintf(f, "feature_%d%s", j, (j + 1 < d) ? "," : "\n");

    /* Pontos. */
    for (long i = 0; i < n; i++) {
        int c = rand() % k;
        for (int j = 0; j < d; j++) {
            double v = gaussian(centers[c * d + j], spread);
            fprintf(f, "%.6g%s", v, (j + 1 < d) ? "," : "\n");
        }
    }

    fclose(f);
    free(centers);
    fprintf(stderr, "gerado %s: %ld pontos, %d dims, %d clusters (seed=%u)\n",
            out, n, d, k, seed);
    return 0;
}
