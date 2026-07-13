# K-means MPI + OpenMP
#
# No macOS, o mpicc do OpenMPI usa o clang (sem suporte nativo a OpenMP),
# então apontamos OMPI_CC para o gcc do Homebrew. No Linux (plataforma da
# disciplina) o mpicc já usa o gcc, que suporta -fopenmp nativamente.

CC      = mpicc
CFLAGS  = -O3 -std=c11 -Wall -Wextra -fopenmp
LDLIBS  = -lm

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
  OMPI_CC ?= gcc-16
  export OMPI_CC
endif

all: kmeans gen_dataset

kmeans: src/main.c src/io.c src/kmeans.c src/io.h src/kmeans.h src/common.h
	$(CC) $(CFLAGS) -o $@ src/main.c src/io.c src/kmeans.c $(LDLIBS)

gen_dataset: tools/gen_dataset.c
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

test: all
	./tests/run_tests.sh

clean:
	rm -f kmeans gen_dataset src/*.o
	rm -rf data
	rm -f clusters.csv centroids.csv

.PHONY: all test clean
