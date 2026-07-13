#ifndef IO_H
#define IO_H

/* Rotinas de arquivo — executadas SOMENTE pelo rank 0.
 * A máquina virtual MPI não compartilha disco: nenhum outro rank abre arquivo.
 * Os dados chegam aos demais ranks por rede (MPI_Scatterv), e os rótulos
 * voltam ao rank 0 por rede (MPI_Gatherv). */

/* Lê um CSV numérico para um vetor row-major (n*d doubles).
 * Detecta automaticamente um cabeçalho (1ª linha não-numérica) e o ignora.
 * Infere d a partir da 1ª linha de dados; aborta (DIE) se alguma linha
 * tiver número de colunas diferente ou valor não numérico.
 * Aloca *out (o chamador deve liberar). Preenche *n e *d. Retorna 0. */
int read_csv(const char *path, double **out, int *n, int *d);

/* Escreve os pontos com seus rótulos.
 * Cabeçalho: feature_0,...,feature_{d-1},cluster */
void write_clusters(const char *path, const double *pts, const int *labels,
                    int n, int d);

/* Escreve os centroides finais e o tamanho de cada cluster.
 * Cabeçalho: cluster_id,centroid_0,...,centroid_{d-1},size */
void write_centroids(const char *path, const double *cent, const int *sizes,
                     int k, int d);

#endif /* IO_H */
