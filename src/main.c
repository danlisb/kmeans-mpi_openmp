/* K-means distribuído com MPI + OpenMP.
 *
 * Uso:
 *   mpirun -np <P> ./kmeans <input.csv> <clusters.csv> <centroids.csv> \
 *          <K> [max_iter=100] [tol=1e-4]
 *   OMP_NUM_THREADS=<T> controla as threads OpenMP.
 *
 * Contorno do disco não-compartilhado: apenas o rank 0 lê o CSV de entrada e
 * escreve os CSVs de saída. Os pontos são distribuídos por rede (MPI_Scatterv)
 * e os rótulos reunidos por rede (MPI_Gatherv). A redução global dos
 * centroides é feita com MPI_Allreduce, deixando todos os ranks idênticos.
 */
#include "common.h"
#include "io.h"
#include "kmeans.h"
#include <string.h>

#define SEED 42u

/* Distância euclidiana ao quadrado entre dois vetores de dimensão d. */
static double sqdist(const double *a, const double *b, int d)
{
    double s = 0.0;
    for (int j = 0; j < d; j++) { double diff = a[j] - b[j]; s += diff * diff; }
    return s;
}

/* Rank 0 escolhe os centroides iniciais por k-means++: o 1º ponto é sorteado
 * uniformemente e cada centroide seguinte é sorteado com probabilidade
 * proporcional ao quadrado da distância ao centroide mais próximo já escolhido.
 * Isso espalha os centroides e evita mínimos locais ruins (ex.: dois centroides
 * no mesmo blob). Determinístico dado o seed. */
static void init_centroids(const double *pts, int n, int d, int k, double *cent)
{
    double *dist2 = malloc((size_t)n * sizeof(double));
    if (!dist2) DIE("sem memoria na inicializacao de centroides");

    srand(SEED);

    /* 1º centroide: uniforme. */
    int first = rand() % n;
    memcpy(&cent[0], &pts[(size_t)first * d], (size_t)d * sizeof(double));
    for (int i = 0; i < n; i++)
        dist2[i] = sqdist(&pts[(size_t)i * d], &cent[0], d);

    /* Demais: proporcional a D(x)^2. */
    for (int c = 1; c < k; c++) {
        double total = 0.0;
        for (int i = 0; i < n; i++) total += dist2[i];

        double target = (rand() / ((double)RAND_MAX + 1.0)) * total;
        int chosen = n - 1;
        double acc = 0.0;
        for (int i = 0; i < n; i++) {
            acc += dist2[i];
            if (acc >= target) { chosen = i; break; }
        }

        memcpy(&cent[(size_t)c * d], &pts[(size_t)chosen * d],
               (size_t)d * sizeof(double));
        for (int i = 0; i < n; i++) {
            double dd = sqdist(&pts[(size_t)i * d], &cent[(size_t)c * d], d);
            if (dd < dist2[i]) dist2[i] = dd;
        }
    }

    free(dist2);
}

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 5) {
        if (rank == 0)
            fprintf(stderr,
                "uso: mpirun -np <P> %s <input.csv> <clusters.csv> "
                "<centroids.csv> <K> [max_iter=100] [tol=1e-4]\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    const char *in_path   = argv[1];
    const char *out_clu   = argv[2];
    const char *out_cen   = argv[3];
    int    K        = atoi(argv[4]);
    int    max_iter = (argc > 5) ? atoi(argv[5]) : 100;
    double tol      = (argc > 6) ? atof(argv[6]) : 1e-4;

    /* --- Rank 0 lê o arquivo (único que toca o disco) --- */
    double *pts_all = NULL;
    int     N = 0, D = 0;
    if (rank == 0) {
        read_csv(in_path, &pts_all, &N, &D);
        if (K < 1)  DIE("K deve ser >= 1");
        if (K > N)  DIE("K (%d) maior que o numero de pontos (%d)", K, N);
    }

    /* --- Difunde metadados (N, D, K) --- */
    Meta meta = { N, D, K };
    MPI_Bcast(&meta, 3, MPI_INT, 0, MPI_COMM_WORLD);
    N = meta.n; D = meta.d; K = meta.k;

    /* --- Centroides iniciais: rank 0 escolhe por k-means++ ---
     * Seeding serial (só no rank 0), feito ANTES de cronometrar — assim como a
     * leitura do arquivo, é pré-processamento e não faz parte do núcleo paralelo
     * cujo speedup queremos medir. É difundido dentro da região cronometrada. */
    double *cent = malloc((size_t)K * D * sizeof(double));
    if (rank == 0) init_centroids(pts_all, N, D, K, cent);

    /* A partir daqui mede-se o núcleo paralelo: scatter + iterações + gather. */
    double t0 = MPI_Wtime();

    /* --- Particiona N pontos entre os ranks (contagens/deslocamentos) --- */
    int *cnt_pts = malloc((size_t)size * sizeof(int));  /* em pontos */
    int *dsp_pts = malloc((size_t)size * sizeof(int));
    int base = N / size, rem = N % size, off = 0;
    for (int r = 0; r < size; r++) {
        int c = base + (r < rem ? 1 : 0);
        cnt_pts[r] = c;
        dsp_pts[r] = off;
        off += c;
    }
    int n_local = cnt_pts[rank];

    /* Scatterv trabalha em elementos (doubles), não em pontos. */
    int *cnt_el = malloc((size_t)size * sizeof(int));
    int *dsp_el = malloc((size_t)size * sizeof(int));
    for (int r = 0; r < size; r++) {
        cnt_el[r] = cnt_pts[r] * D;
        dsp_el[r] = dsp_pts[r] * D;
    }

    double *pts_local = malloc((size_t)n_local * D * sizeof(double));
    MPI_Scatterv(pts_all, cnt_el, dsp_el, MPI_DOUBLE,
                 pts_local, n_local * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* Difunde os centroides iniciais a todos os ranks. */
    MPI_Bcast(cent, K * D, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /* --- Loop de Lloyd --- */
    int   *labels_local = malloc((size_t)n_local * sizeof(int));
    double *sum   = malloc((size_t)K * D * sizeof(double));
    double *gsum  = malloc((size_t)K * D * sizeof(double));
    long   *count  = malloc((size_t)K * sizeof(long));
    long   *gcount = malloc((size_t)K * sizeof(long));

    int    it;
    double shift = 0.0;
    for (it = 0; it < max_iter; it++) {
        /* Passo de atribuição paralelo (OpenMP) sobre os pontos locais. */
        assign_and_accumulate(pts_local, n_local, D, K, cent,
                              labels_local, sum, count);

        /* Redução global (MPI): somas e contagens por cluster. */
        MPI_Allreduce(sum,   gsum,   K * D, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(count, gcount, K,     MPI_LONG,   MPI_SUM, MPI_COMM_WORLD);

        /* Novos centroides idênticos em todos os ranks. */
        shift = update_centroids(cent, gsum, gcount, K, D);
        if (shift < tol) { it++; break; }
    }

    /* --- Reúne os rótulos no rank 0 (Gatherv, em pontos) --- */
    int *labels_all = NULL;
    if (rank == 0) labels_all = malloc((size_t)N * sizeof(int));
    MPI_Gatherv(labels_local, n_local, MPI_INT,
                labels_all, cnt_pts, dsp_pts, MPI_INT, 0, MPI_COMM_WORLD);

    double t1 = MPI_Wtime();

    /* --- Rank 0 escreve as saídas --- */
    if (rank == 0) {
        int *sizes = malloc((size_t)K * sizeof(int));
        for (int c = 0; c < K; c++) sizes[c] = (int)gcount[c];

        write_clusters(out_clu, pts_all, labels_all, N, D);
        write_centroids(out_cen, cent, sizes, K, D);

        printf("K-means concluido: N=%d D=%d K=%d | iteracoes=%d | "
               "shift_final=%.3e | tempo=%.4fs | procs=%d\n",
               N, D, K, it, shift, t1 - t0, size);

        free(sizes);
        free(labels_all);
        free(pts_all);
    }

    free(cnt_pts); free(dsp_pts); free(cnt_el); free(dsp_el);
    free(pts_local); free(cent); free(labels_local);
    free(sum); free(gsum); free(count); free(gcount);

    MPI_Finalize();
    return 0;
}
