#define _POSIX_C_SOURCE 200809L
#include "io.h"
#include "common.h"
#include <string.h>
#include <ctype.h>

/* Verdadeiro se a linha só contém espaços em branco. */
static int is_blank(const char *s)
{
    for (; *s; s++)
        if (!isspace((unsigned char)*s)) return 0;
    return 1;
}

/* Conta campos separados por vírgula numa linha não vazia. */
static int count_fields(const char *s)
{
    int fields = 1;
    for (; *s && *s != '\n' && *s != '\r'; s++)
        if (*s == ',') fields++;
    return fields;
}

/* Tenta interpretar a linha como 'max' doubles separados por vírgula.
 * Retorna a quantidade lida; define *ok=0 se algum campo não for numérico. */
static int parse_fields(const char *line, double *vals, int max, int *ok)
{
    *ok = 1;
    int cnt = 0;
    const char *p = line;
    while (*p && *p != '\n' && *p != '\r' && cnt < max) {
        while (*p == ' ' || *p == '\t') p++;
        char *end;
        double v = strtod(p, &end);
        if (end == p) { *ok = 0; return cnt; }   /* não é número */
        vals[cnt++] = v;
        p = end;
        while (*p == ' ' || *p == '\t') p++;
        if (*p == ',') p++;
    }
    return cnt;
}

int read_csv(const char *path, double **out, int *n, int *d)
{
    FILE *f = fopen(path, "r");
    if (!f) DIE("nao foi possivel abrir o arquivo de entrada: %s", path);

    char  *line = NULL;
    size_t cap  = 0;
    ssize_t len;
    long   lineno = 0;

    int    D = -1;
    long   cnt = 0, alloc = 0;
    double *buf = NULL;
    int    header_checked = 0;

    while ((len = getline(&line, &cap, f)) != -1) {
        lineno++;
        if (is_blank(line)) continue;

        int fields = count_fields(line);
        double *tmp = malloc((size_t)fields * sizeof(double));
        if (!tmp) DIE("sem memoria ao ler CSV");

        int ok;
        int got = parse_fields(line, tmp, fields, &ok);

        /* A primeira linha não numérica é tratada como cabeçalho e ignorada. */
        if (!header_checked) {
            header_checked = 1;
            if (!ok) { free(tmp); continue; }
        }
        if (!ok) { free(tmp); DIE("linha %ld: valor nao numerico", lineno); }

        if (D < 0) {
            D = got;
        } else if (got != D) {
            free(tmp);
            DIE("linha %ld: esperado %d colunas, encontrado %d", lineno, D, got);
        }

        if (cnt + 1 > alloc) {
            alloc = alloc ? alloc * 2 : 1024;
            buf = realloc(buf, (size_t)alloc * (size_t)D * sizeof(double));
            if (!buf) DIE("sem memoria ao ler CSV");
        }
        memcpy(&buf[(size_t)cnt * D], tmp, (size_t)D * sizeof(double));
        cnt++;
        free(tmp);
    }

    free(line);
    fclose(f);

    if (cnt == 0) DIE("arquivo de entrada sem dados: %s", path);

    *out = buf;
    *n   = (int)cnt;
    *d   = D;
    return 0;
}

void write_clusters(const char *path, const double *pts, const int *labels,
                    int n, int d)
{
    FILE *f = fopen(path, "w");
    if (!f) DIE("nao foi possivel escrever %s", path);

    for (int j = 0; j < d; j++) fprintf(f, "feature_%d,", j);
    fprintf(f, "cluster\n");

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < d; j++)
            fprintf(f, "%.6g,", pts[(size_t)i * d + j]);
        fprintf(f, "%d\n", labels[i]);
    }
    fclose(f);
}

void write_centroids(const char *path, const double *cent, const int *sizes,
                     int k, int d)
{
    FILE *f = fopen(path, "w");
    if (!f) DIE("nao foi possivel escrever %s", path);

    fprintf(f, "cluster_id,");
    for (int j = 0; j < d; j++) fprintf(f, "centroid_%d,", j);
    fprintf(f, "size\n");

    for (int c = 0; c < k; c++) {
        fprintf(f, "%d,", c);
        for (int j = 0; j < d; j++)
            fprintf(f, "%.6g,", cent[(size_t)c * d + j]);
        fprintf(f, "%d\n", sizes[c]);
    }
    fclose(f);
}
