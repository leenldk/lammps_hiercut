#!/bin/bash
#set -x

spack env activate umi
spack load intel-oneapi-mkl
spack load cuda@12.4.0
spack load openmpi@5.0.6

PREFIX=$(date "+%m_%d_%H%M%S")

EXE=$PWD/../../../build_kokkos/lmp

INPUT=in.lj

OUT_FILE=$PREFIX.out

echo $(which mpirun)
# 1 GPU
# CUDA_VISIBLE_DEVICES=0 OMP_PROC_BIND=spread OMP_PLACES=threads mpirun --bind-to core --map-by socket:PE=10 -np 1 $EXE -k on t 10 g 1 -sf kk -in $INPUT 2>&1 | tee $OUT_FILE
# 2 GPU
CUDA_VISIBLE_DEVICES=0,1 OMP_PROC_BIND=spread OMP_PLACES=threads mpirun --bind-to core --map-by socket:PE=10 -np 2 $EXE -k on t 10 g 2 -sf kk -in $INPUT 2>&1 | tee $OUT_FILE
# 4 GPU
# CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_PROC_BIND=spread OMP_PLACES=threads mpirun --bind-to core --map-by socket:PE=10 -np 4 $EXE -k on t 10 g 4 -sf kk -in $INPUT 2>&1 | tee $OUT_FILE
