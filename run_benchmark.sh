#!/usr/bin/env bash
#
# Mede o speedup do K-means variando processos MPI (-np) e threads OpenMP.
# O tempo reportado é o do núcleo paralelo (scatter + iterações + gather),
# medido internamente com MPI_Wtime, sem contar a leitura/escrita de arquivo.
#
# Variáveis de ambiente: N, D, K, MAX_PROCS, MAX_THREADS (têm padrões).
# Uso: ./run_benchmark.sh

set -u
cd "$(dirname "$0")"
mkdir -p data

N=${N:-1000000}
D=${D:-16}
K=${K:-24}
SPREAD=${SPREAD:-2}

DATA="data/bench.csv"
if [ ! -f "$DATA" ]; then
    echo "Gerando dataset de benchmark (N=$N D=$D K=$K spread=$SPREAD)..."
    ./gen_dataset "$N" "$D" "$K" "$DATA" 42 "$SPREAD" >/dev/null 2>&1
fi

run() {  # $1=np  $2=threads  -> imprime o tempo em segundos
    OMP_NUM_THREADS="$2" mpirun -np "$1" ./kmeans "$DATA" \
        /tmp/bench_clu.csv /tmp/bench_cen.csv "$K" 100 1e-4 2>/dev/null \
        | sed -n 's/.*tempo=\([0-9.]*\)s.*/\1/p'
}

echo "Dataset: N=$N  D=$D  K=$K   (nucleos disponiveis: $(getconf _NPROCESSORS_ONLN 2>/dev/null || echo '?'))"
echo

base=$(run 1 1)
printf "%-8s %-9s %-12s %-9s\n" "procs" "threads" "tempo(s)" "speedup"
printf "%-8s %-9s %-12s %-9s\n" "1" "1" "$base" "1.00"

for np in 1 2 4; do
    for t in 1 2 4; do
        [ "$np" = "1" ] && [ "$t" = "1" ] && continue
        tm=$(run "$np" "$t")
        sp=$(awk -v b="$base" -v x="$tm" 'BEGIN { if (x+0>0) printf "%.2f", b/x; else printf "-" }')
        printf "%-8s %-9s %-12s %-9s\n" "$np" "$t" "$tm" "$sp"
    done
done

echo
echo "Observacao: procs*threads acima do numero de nucleos causa sobre-subscricao"
echo "e reduz o ganho; os melhores numeros ficam com procs*threads <= nucleos."
