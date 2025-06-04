#!/bin/bash
#set -x

source ../../env.sh

PREFIX=$(date "+%m_%d_%H%M%S")

EXE=$PWD/../../../build_kokkos/lmp

INPUT=in.lj

OUT_FILE=$PREFIX.out

echo $(which mpirun)
CUDA_VISIBLE_DEVICES=2 OMP_PROC_BIND=spread OMP_PLACES=threads mpirun --bind-to core --map-by socket:PE=20 -np 1 $EXE -k on t 20 g 1 -sf kk -in $INPUT 2>&1 | tee $OUT_FILE
