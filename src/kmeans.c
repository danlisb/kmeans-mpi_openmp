#include "kmeans.h"
#include <stdlib.h>
#include <math.h>

void assign_and_accumulate(const double *pts, int n_local, int d, int k,
                           const double *cent, int *labels,
                           double *sum, long *count)
{
    for (int i = 0; i < k * d; i++) sum[i] = 0.0;
    for (int c = 0; c < k; c++)     count[c] = 0;

    #pragma omp parallel
    {
        /* Buffers privados por thread: evitam contenção no laço quente.
         * São mesclados nos globais uma única vez, na região crítica. */
        double *lsum = calloc((size_t)k * d, sizeof(double));
        long   *lcnt = calloc((size_t)k, sizeof(long));

        #pragma omp for schedule(static)
        for (int i = 0; i < n_local; i++) {
            const double *p = &pts[(size_t)i * d];

            int    best = 0;
            double best_dist = INFINITY;
            for (int c = 0; c < k; c++) {
                const double *ct = &cent[(size_t)c * d];
                double dist = 0.0;
                for (int j = 0; j < d; j++) {
                    double diff = p[j] - ct[j];
                    dist += diff * diff;
                }
                if (dist < best_dist) { best_dist = dist; best = c; }
            }

            labels[i] = best;
            double *ls = &lsum[(size_t)best * d];
            for (int j = 0; j < d; j++) ls[j] += p[j];
            lcnt[best]++;
        }

        #pragma omp critical
        {
            for (int i = 0; i < k * d; i++) sum[i]   += lsum[i];
            for (int c = 0; c < k; c++)     count[c] += lcnt[c];
        }

        free(lsum);
        free(lcnt);
    }
}

double update_centroids(double *cent, const double *gsum, const long *gcount,
                        int k, int d)
{
    double max_shift = 0.0;

    for (int c = 0; c < k; c++) {
        if (gcount[c] == 0) continue;  /* cluster vazio: mantém centroide */

        double shift = 0.0;
        for (int j = 0; j < d; j++) {
            double nv   = gsum[(size_t)c * d + j] / (double)gcount[c];
            double diff = nv - cent[(size_t)c * d + j];
            shift += diff * diff;
            cent[(size_t)c * d + j] = nv;
        }
        shift = sqrt(shift);
        if (shift > max_shift) max_shift = shift;
    }

    return max_shift;
}
