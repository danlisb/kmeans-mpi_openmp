# Clustering de grandes conjuntos de dados com K-means usando MPI + OpenMP

> **Disciplina:** Programação Paralela e Distribuída (MPI + OpenMP)
> **Tema:** Clustering de grandes conjuntos de dados com K-means
> **Autor:** _(preencher)_
> **Data:** julho de 2026

---

## 1. Introdução

O agrupamento (*clustering*) é uma tarefa fundamental de aprendizado não
supervisionado: dado um conjunto de registros, deseja-se particioná-los em
grupos (*clusters*) de forma que registros semelhantes fiquem no mesmo grupo. O
algoritmo **K-means** é o método mais difundido para essa tarefa, mas seu custo
cresce linearmente com o número de pontos, de dimensões e de clusters, tornando
o processamento de **grandes conjuntos de dados** caro em uma única máquina/thread.

Este trabalho implementa o K-means explorando **dois níveis de paralelismo**:

- **MPI** (memória distribuída) para dividir os registros entre processos, que
  podem estar em nós diferentes, e para realizar a **redução global dos
  centroides** a cada iteração;
- **OpenMP** (memória compartilhada) para paralelizar, dentro de cada processo,
  o **cálculo das distâncias** entre pontos e centroides — a parte mais cara do
  algoritmo.

Um requisito prático do ambiente-alvo é que **os nós MPI não compartilham
disco**. Isso é tratado explicitamente pela arquitetura do programa, descrita na
Seção 4.

## 2. Fundamentação teórica

### 2.1. O algoritmo K-means (Lloyd)

Dado um conjunto de `N` pontos em `R^D` e um número `K` de clusters, o K-means
busca `K` centroides que minimizem a soma das distâncias quadráticas de cada
ponto ao seu centroide mais próximo (a *inércia*). O algoritmo de Lloyd itera
dois passos até a convergência:

1. **Atribuição:** cada ponto é associado ao centroide mais próximo (distância
   euclidiana).
2. **Atualização:** cada centroide é recalculado como a média dos pontos que lhe
   foram atribuídos.

O custo por iteração é `O(N · K · D)`, dominado pelo passo de atribuição.

### 2.2. Inicialização k-means++

A qualidade do resultado depende fortemente dos centroides iniciais. A
inicialização aleatória simples pode colocar dois centroides no mesmo grupo,
levando a um mínimo local ruim. O **k-means++** escolhe o primeiro centroide
uniformemente e cada centroide seguinte com probabilidade proporcional ao
**quadrado da distância** ao centroide mais próximo já escolhido, espalhando as
sementes e melhorando muito a qualidade e a velocidade de convergência.

### 2.3. MPI e OpenMP

**MPI** (*Message Passing Interface*) é o padrão para paralelismo em **memória
distribuída**: processos independentes, possivelmente em nós distintos,
comunicam-se por troca de mensagens (`MPI_Scatterv`, `MPI_Allreduce`, etc.).

**OpenMP** é o padrão para paralelismo em **memória compartilhada**: diretivas
`#pragma omp` criam threads que compartilham a memória do processo, ideal para
paralelizar laços.

A combinação (**modelo híbrido**) explora a hierarquia real das máquinas atuais:
MPI entre nós/processos e OpenMP entre os núcleos de cada nó.

## 3. O problema do disco não-compartilhado

No ambiente MPI-alvo, cada processo pode executar em um nó com seu **próprio
sistema de arquivos**. Um arquivo visível ao rank 0 pode não existir nos demais
nós. Assim, a estratégia ingênua — cada processo abre o CSV e lê sua fatia —
**falharia**, pois só um dos nós enxerga o arquivo.

**Solução adotada:** concentrar **todo** o acesso a disco no rank 0.

- Somente o rank 0 lê o CSV de entrada e escreve os CSVs de saída.
- Os dados são distribuídos aos demais processos **pela rede** (`MPI_Scatterv`);
  os rótulos resultantes retornam ao rank 0 pela rede (`MPI_Gatherv`).
- Nenhum outro processo executa operações de arquivo.

Essa decisão isola a dependência de disco em um único ponto e torna o programa
correto independentemente de o sistema de arquivos ser compartilhado ou não.

## 4. Arquitetura e implementação

O código está organizado em módulos com responsabilidades bem definidas:

| Arquivo | Responsabilidade |
|---|---|
| `src/main.c` | Orquestração MPI: partição, `Scatterv`, `Allreduce`, `Gatherv`, loop |
| `src/kmeans.c` | Atribuição (OpenMP), atualização de centroides, distâncias |
| `src/io.c` | Leitura/escrita de CSV (**apenas rank 0**) |
| `tools/gen_dataset.c` | Geração de datasets sintéticos (blobs gaussianos) |

### 4.1. Fluxo de execução

1. O **rank 0** lê o CSV para a memória (`N×D`) e escolhe `K` centroides
   iniciais por k-means++.
2. `MPI_Bcast` difunde `N`, `D`, `K` e os centroides iniciais.
3. `MPI_Scatterv` distribui os pontos: cada rank recebe um bloco de
   aproximadamente `N/P` pontos (contagens e deslocamentos calculados para
   dividir o resto de forma equilibrada).
4. **Loop de Lloyd**, executado por todos os ranks:
   - **Atribuição (OpenMP):** para cada ponto local, calcula-se a distância
     euclidiana ao quadrado a cada centroide e escolhe-se o menor. O laço é
     paralelizado com `#pragma omp parallel for`; cada thread acumula somas e
     contagens por cluster em **buffers privados**, fundidos ao final numa
     região crítica — evitando contenção no laço quente.
   - **Redução global (MPI):** `MPI_Allreduce` com `MPI_SUM` soma, entre todos
     os processos, as coordenadas acumuladas (`K×D`) e as contagens (`K`).
   - **Atualização:** cada rank divide soma por contagem, obtendo centroides
     **idênticos** em todos os processos, e mede o deslocamento máximo.
   - **Convergência:** encerra quando o deslocamento máximo `< tol`.
5. `MPI_Gatherv` reúne os rótulos no rank 0, que escreve `clusters.csv` e
   `centroids.csv`.

### 4.2. Por que `MPI_Allreduce`

Poder-se-ia usar `MPI_Reduce` (concentrando a soma no rank 0) seguido de
`MPI_Bcast` dos novos centroides. Optou-se por `MPI_Allreduce` porque ele já
devolve o resultado a **todos** os ranks: cada processo recalcula os centroides
e avalia a convergência localmente, com o mesmo resultado, sem uma etapa extra
de broadcast. Como todos partem dos mesmos dados reduzidos, a execução é
determinística e o resultado independe do número de processos.

### 4.3. Casos de borda

- **Cluster vazio** (nenhum ponto atribuído numa iteração): o centroide anterior
  é mantido.
- **Validação de entrada:** falha de abertura de arquivo, CSV malformado (com o
  número da linha), `K < 1` ou `K > N` abortam a execução com `MPI_Abort` e uma
  mensagem clara.
- **Reprodutibilidade:** a inicialização usa semente fixa, garantindo resultados
  idênticos entre execuções.

## 5. Metodologia experimental

- **Máquina:** CPU de 10 núcleos.
- **Toolchain:** OpenMPI 5 + GCC com `-O3 -fopenmp`.
- **Dataset:** `N = 1.000.000` pontos, `D = 16` dimensões, `K = 24` clusters,
  gerado sinteticamente (`spread = 2`, convergindo em ~38 iterações).
- **Medida:** tempo do **núcleo paralelo** (distribuição + iterações + coleta),
  obtido com `MPI_Wtime`. A leitura/escrita de arquivo e o seeding serial ficam
  **fora** da medição, pois são pré-processamento e não fazem parte da porção
  paralelizável cujo *speedup* se deseja avaliar.
- Variou-se o número de processos MPI (`P ∈ {1, 2, 4}`) e de threads OpenMP
  (`T ∈ {1, 2, 4}`). O *speedup* é `tempo(1,1) / tempo(P,T)`.

## 6. Resultados

| Processos (P) | Threads (T) | Tempo (s) | Speedup |
|:---:|:---:|:---:|:---:|
| 1 | 1 | 14.11 | 1.00 |
| 1 | 2 |  7.31 | 1.93 |
| 1 | 4 |  3.99 | 3.54 |
| 2 | 1 |  7.46 | 1.89 |
| 2 | 2 |  4.21 | 3.35 |
| 2 | 4 |  2.64 | 5.35 |
| 4 | 1 |  4.33 | 3.26 |
| 4 | 2 |  2.69 | 5.24 |
| 4 | 4 |  2.62 | 5.38 |

### Análise

- **Escalabilidade OpenMP** (fixando `P=1`): de 1 para 2 threads o *speedup* é
  1.93× (quase ideal) e de 1 para 4 threads é 3.54×. O passo de atribuição é
  altamente paralelizável, com pouca sincronização (apenas a fusão dos buffers
  por thread).
- **Escalabilidade MPI** (fixando `T=1`): 1.89× com 2 processos e 3.26× com 4
  processos. O ganho é ligeiramente menor que o do OpenMP porque o `Allreduce`
  introduz comunicação a cada iteração.
- **Modelo híbrido:** as melhores marcas vêm da combinação — `2×4` e `4×2`
  atingem ~5.3×, aproveitando 8 vias de paralelismo dentro dos 10 núcleos.
- **Sobre-subscrição:** `4×4` (16 threads em 10 núcleos) praticamente não melhora
  em relação a `4×2`, pois passa a haver mais threads que núcleos físicos — em
  linha com a **Lei de Amdahl** e com a contenção por recursos.

O experimento confirma que **ambos** os níveis de paralelismo contribuem e que a
combinação MPI + OpenMP é a mais eficiente, como esperado do modelo híbrido.

## 7. Conclusão

Implementou-se o K-means para grandes conjuntos de dados com paralelismo
híbrido: MPI distribuindo os registros e reduzindo globalmente os centroides, e
OpenMP paralelizando o cálculo das distâncias locais. A restrição de **disco não
compartilhado** foi resolvida centralizando todo o I/O no rank 0 e distribuindo
os dados pela rede com `Scatterv`/`Gatherv`. Os experimentos mostraram *speedup*
de até **5.4×** em uma máquina de 10 núcleos, com escalabilidade consistente
tanto em MPI quanto em OpenMP, e a validação confirmou que a versão paralela
produz **exatamente o mesmo resultado** da serial.

## Referências

1. Lloyd, S. P. *Least squares quantization in PCM*. IEEE Transactions on
   Information Theory, 1982.
2. Arthur, D.; Vassilvitskii, S. *k-means++: The Advantages of Careful Seeding*.
   SODA, 2007.
3. MPI Forum. *MPI: A Message-Passing Interface Standard*.
4. OpenMP Architecture Review Board. *OpenMP Application Programming Interface*.
