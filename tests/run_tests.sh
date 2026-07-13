#!/usr/bin/env bash
#
# Testes de correção do K-means MPI + OpenMP.
#   A) tiny.csv: dois grupos óbvios devem ser separados corretamente.
#   B) serial vs. paralelo: (np=1,T=1) e (np=4,T=4) produzem rótulos idênticos.
#   C) determinismo: duas execuções iguais produzem saída idêntica.
#
# Requer os binários já compilados (rode via `make test`).

set -u
cd "$(dirname "$0")/.."

PASS=0
FAIL=0
TMP="tests/.tmp"
mkdir -p "$TMP" data

pass() { echo "  [PASS] $1"; PASS=$((PASS + 1)); }
fail() { echo "  [FALHOU] $1"; FAIL=$((FAIL + 1)); }

# --- Teste A: separação em tiny.csv ---
echo "Teste A: separacao de 2 grupos obvios (tiny.csv)"
mpirun -np 1 ./kmeans tests/tiny.csv "$TMP/A_clu.csv" "$TMP/A_cen.csv" 2 >/dev/null 2>&1
res=$(tail -n +2 "$TMP/A_clu.csv" | awk -F, '
    NR<=3 { g1[$NF]=1 } NR>3 { g2[$NF]=1 }
    END {
        n1=0; for (x in g1) n1++;
        n2=0; for (x in g2) n2++;
        overlap=0; for (x in g1) if (x in g2) overlap=1;
        print (n1==1 && n2==1 && overlap==0) ? "OK" : "BAD";
    }')
[ "$res" = "OK" ] && pass "pontos 0-2 e 3-5 em clusters distintos" \
                   || fail "grupos nao separados corretamente (res=$res)"

# --- Teste B: serial vs. paralelo ---
echo "Teste B: serial (np=1,T=1) vs. paralelo (np=4,T=4) -> rotulos identicos"
./gen_dataset 5000 3 4 "$TMP/mid.csv" 7 >/dev/null 2>&1
OMP_NUM_THREADS=1 mpirun -np 1 ./kmeans "$TMP/mid.csv" "$TMP/B_s.csv" "$TMP/B_sc.csv" 4 >/dev/null 2>&1
OMP_NUM_THREADS=4 mpirun -np 4 ./kmeans "$TMP/mid.csv" "$TMP/B_p.csv" "$TMP/B_pc.csv" 4 >/dev/null 2>&1
if diff -q "$TMP/B_s.csv" "$TMP/B_p.csv" >/dev/null 2>&1; then
    pass "saidas identicas (paralelizacao preserva a correcao)"
else
    fail "saidas diferentes entre serial e paralelo"
fi

# --- Teste C: determinismo ---
echo "Teste C: duas execucoes iguais -> saida identica"
OMP_NUM_THREADS=2 mpirun -np 2 ./kmeans "$TMP/mid.csv" "$TMP/C_1.csv" "$TMP/C_1c.csv" 4 >/dev/null 2>&1
OMP_NUM_THREADS=2 mpirun -np 2 ./kmeans "$TMP/mid.csv" "$TMP/C_2.csv" "$TMP/C_2c.csv" 4 >/dev/null 2>&1
if diff -q "$TMP/C_1.csv" "$TMP/C_2.csv" >/dev/null 2>&1; then
    pass "resultado reprodutivel (seed fixo)"
else
    fail "resultado nao reprodutivel"
fi

rm -rf "$TMP"
echo "-----------------------------------------"
echo "Resultado: $PASS passaram, $FAIL falharam"
[ "$FAIL" -eq 0 ]
