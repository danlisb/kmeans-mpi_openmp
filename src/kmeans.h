#ifndef KMEANS_H
#define KMEANS_H

/* Núcleo do K-means — funções LOCAIS a cada rank (não contêm chamadas MPI).
 * A paralelização com OpenMP acontece aqui, no passo de atribuição. */

/* Passo de atribuição + acumulação, paralelizado com OpenMP.
 * Para cada ponto local, encontra o centroide mais próximo (distância
 * euclidiana ao quadrado) e grava seu índice em labels[i].
 * Acumula, por cluster, a soma das coordenadas (sum[k*d]) e a contagem
 * (count[k]) — que serão reduzidos globalmente via MPI pelo chamador.
 * sum e count são zerados no início. */
void assign_and_accumulate(const double *pts, int n_local, int d, int k,
                           const double *cent, int *labels,
                           double *sum, long *count);

/* Recalcula os centroides a partir das somas/contagens GLOBAIS.
 * Cluster vazio (gcount==0): mantém o centroide anterior.
 * Retorna o maior deslocamento (euclidiano) de centroide — usado para
 * testar convergência. */
double update_centroids(double *cent, const double *gsum, const long *gcount,
                        int k, int d);

#endif /* KMEANS_H */
