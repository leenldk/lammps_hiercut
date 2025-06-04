#!/bin/bash
spack env activate umi
spack load intel-oneapi-mkl
spack load cuda@12.8
spack load openmpi@5.0.6
