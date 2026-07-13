# K-means distribuído com MPI + OpenMP

Clustering de grandes conjuntos de dados com o algoritmo **K-means** (Lloyd),
paralelizado em dois níveis:

- **MPI** — distribui os registros entre os processos e faz a **redução global
  dos centroides** a cada iteração;
- **OpenMP** — paraleliza o **cálculo das distâncias locais** dentro de cada
  processo.

A entrada e a saída são arquivos **CSV**. O programa foi escrito para rodar em um
ambiente MPI onde **os nós não compartilham disco** (ver seção
[Contorno do disco não-compartilhado](#contorno-do-disco-não-compartilhado)).

---

## Requisitos

- Uma implementação de MPI (ex.: **OpenMPI** ou **MPICH**) com `mpicc` e `mpirun`.
- Um compilador C com suporte a **OpenMP** (`-fopenmp`) — o `gcc` já traz.
- `make`.

### macOS

No macOS, o `mpicc` do OpenMPI usa o `clang`, que não traz o runtime de OpenMP.
Instale o GCC do Homebrew e o OpenMPI:

```bash
brew install open-mpi gcc
```

O `Makefile` detecta o macOS automaticamente e aponta `OMPI_CC` para o `gcc` do
Homebrew (por padrão `gcc-16`; ajuste a versão no `Makefile` se necessário).
Em Linux nenhum ajuste é preciso.

---

## Compilação

```bash
make
```

Gera dois binários:

- `kmeans` — o programa principal (MPI + OpenMP);
- `gen_dataset` — gerador de datasets sintéticos para teste/benchmark.

---

## Uso

### 1. Gerar um dataset de exemplo

```bash
./gen_dataset <n_pontos> <n_dims> <k_clusters> <saida.csv> [seed=42] [spread=1.0]
# exemplo:
./gen_dataset 100000 4 5 data/entrada.csv
```

O gerador cria `k` blobs gaussianos. `spread` controla a dispersão (valores
maiores deixam os grupos mais sobrepostos e o K-means mais difícil).

### 2. Executar o K-means

```bash
mpirun -np <P> ./kmeans <entrada.csv> <clusters.csv> <centroids.csv> \
       <K> [max_iter=100] [tol=1e-4]
```

O número de threads OpenMP por processo é controlado por `OMP_NUM_THREADS`:

```bash
# 4 processos MPI, 4 threads OpenMP cada:
OMP_NUM_THREADS=4 mpirun -np 4 ./kmeans data/entrada.csv clusters.csv centroids.csv 5
```

### Formatos de arquivo

**Entrada** — CSV numérico, uma linha por ponto, `d` colunas. Um cabeçalho é
detectado e ignorado automaticamente (primeira linha não numérica). Exemplo:

```
feature_0,feature_1
14.8961,15.836
45.518,47.0365
```

**`clusters.csv`** — cada ponto de entrada com o cluster atribuído:

```
feature_0,feature_1,cluster
14.8961,15.836,0
45.518,47.0365,2
```

**`centroids.csv`** — os centroides finais e o tamanho de cada cluster:

```
cluster_id,centroid_0,centroid_1,size
0,14.9864,16.9558,678
```

---

## Como funciona

```
rank 0 (ÚNICO que acessa o disco)
  │ lê entrada.csv → matriz N×D em memória
  │ escolhe K centroides iniciais (k-means++)
  │ Bcast(N, D, K, centroides)
  │ Scatterv(pontos) ─────────────►  cada rank recebe seu bloco local
  ▼
LOOP (em TODOS os ranks) até convergir ou max_iter:
  1. Atribuição [OpenMP]: para cada ponto local, distância aos K centroides
     → #pragma omp parallel for, com buffers por thread (sem contenção)
  2. Redução local: somas por cluster (K×D) e contagens (K)
  3. Redução global [MPI]: MPI_Allreduce(SUM) → centroides novos idênticos
     em todos os ranks
  4. Convergência: deslocamento máximo dos centroides < tol
  ▼
rank 0:
  Gatherv(rótulos) ◄─────────────  cada rank envia os rótulos do seu bloco
  escreve clusters.csv e centroids.csv
```

- **Atribuição (OpenMP):** o laço sobre os pontos locais é paralelizado com
  `#pragma omp parallel for`. Cada thread acumula em buffers privados
  (`sum[K*D]`, `count[K]`) e só no fim funde os parciais, evitando contenção.
- **Redução global (MPI):** `MPI_Allreduce` com `MPI_SUM` soma, entre todos os
  processos, as coordenadas e as contagens por cluster. Como o `Allreduce`
  devolve o resultado a todos, cada rank recalcula centroides **idênticos** e
  testa a convergência de forma consistente, sem broadcast adicional.
- **Inicialização (k-means++):** o rank 0 escolhe os centroides iniciais
  proporcionalmente ao quadrado da distância, o que espalha as sementes e evita
  mínimos locais ruins. É determinístico (seed fixo).

### Contorno do disco não-compartilhado

Em muitos ambientes MPI (clusters, VMs) **os nós não compartilham o mesmo
sistema de arquivos**: um arquivo aberto no nó do rank 0 pode simplesmente não
existir nos demais nós. Se cada processo tentasse abrir o CSV, a execução
falharia (ou leria dados diferentes).

A solução adotada é **centralizar todo o acesso a disco no rank 0**:

1. Apenas o rank 0 chama `fopen`/`fread` para ler a entrada e `fopen`/`fwrite`
   para gravar as saídas (ver `src/io.c`).
2. Os dados são entregues aos demais processos **pela rede**, via
   `MPI_Scatterv`; os rótulos voltam ao rank 0 via `MPI_Gatherv`.
3. Nenhum outro rank toca o disco — portanto o programa funciona mesmo sem
   sistema de arquivos compartilhado.

---

## Testes

```bash
make test
```

Executa `tests/run_tests.sh`, que verifica:

- **A** — dois grupos óbvios (`tests/tiny.csv`) são separados corretamente;
- **B** — as execuções **serial** (`np=1, T=1`) e **paralela** (`np=4, T=4`)
  produzem rótulos **idênticos** (a paralelização não altera o resultado);
- **C** — duas execuções iguais produzem saída idêntica (determinismo).

---

## Benchmark de speedup

```bash
./run_benchmark.sh
```

Gera um dataset grande e mede o tempo do núcleo paralelo variando `-np`
(processos MPI) e `OMP_NUM_THREADS` (threads), imprimindo uma tabela de speedup.
Parametrizável por variáveis de ambiente, por exemplo:

```bash
N=2000000 D=8 K=16 ./run_benchmark.sh
```

---

## Estrutura do projeto

```
src/main.c        Orquestração MPI: partição, Scatterv, Allreduce, Gatherv
src/kmeans.c/.h   Atribuição (OpenMP), atualização de centroides, distâncias
src/io.c/.h       Leitura/escrita de CSV (somente rank 0)
src/common.h      Tipos e utilitários compartilhados
tools/gen_dataset.c   Gerador de datasets sintéticos
tests/            Dataset minúsculo e script de testes de correção
run_benchmark.sh  Medição de speedup
Makefile          Alvos: all, kmeans, gen_dataset, test, clean
docs/relatorio.md Relatório
```
