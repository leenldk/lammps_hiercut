Source code of HierCut based on LAMMPS.

# File Structure

The majority of source code is implemented under `src/KOKKOSS`.

`verfication` folder contains input and scripts used for result verfication.

# Setup Environment

Compilation is performed using the following software: 
- \item GCC 12.2.0 (https://gcc.gnu.org/), 
\item CMake 3.27.9 (https://cmake.org/), 
\item CUDA 12.8 (https://developer.nvidia.com/cuda-toolkit), 
\item OpenMPI 5.0.6 (https://www.open-mpi.org/), 
\item Intel oneAPI MKL 2024.2.2 (https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html). 

# Test Cases

Two benchmarks are used for evaluation. The LJ case (https://github.com/lammps/lammps/blob/develop/bench/in.lj) is part of the LAMMPS benchmark suite. 

The 6ZFO case is obtained from the CHARMM benchmark archive (https://www.charmm-gui.org/?doc=archive\&lib=covid19); it is converted to a LAMMPS input format using the input generator provided by CHARMM-GUI, and further processed by LAMMPS to produce the final simulation input. 

# Install and Development

First, ensure that the environment variables for all software dependencies are correctly configured. To compile the source code, execute the following commands in the project directory:

```
mkdir build_kokkos
cd build_kokkos
cmake -DCMAKE_VERBOSE_MAKEFILE=ON -DKokkos_ARCH_ICX=ON \
-DKokkos_ENABLE_CUDA_UVM=ON -DKokkos_ENABLE_OPENMP=ON  \
-DFFT=MKL  -DKokkos_ENABLE_DEBUG=ON -DPKG_EXTRA-DUMP=ON \
-DPKG_COLVARS=ON -C ../cmake/presets/basic.cmake -C \
../cmake/presets/kokkos-cuda.cmake   ../cmake 
make -j20
```

Each result can be reproduced by executing the `run.sh` script located in the corresponding subfolder under `verification`. The simulation outputs are saved in the same folder as the `run.sh` script, with filenames based on timestamps, such as `04_09_104006.out`. The outputs from previous runs, which are used to generate the figures, are also preserved in the corresponding folders.
