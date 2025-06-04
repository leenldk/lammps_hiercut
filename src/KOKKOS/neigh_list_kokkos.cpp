// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#include "neigh_list_kokkos.h"
#include "kokkos.h"
#include "memory_kokkos.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
NeighListKokkos<DeviceType>::NeighListKokkos(class LAMMPS *lmp):NeighList(lmp)
{
  _stride = 1;
  maxneighs = 16;
  maxneighs_outer = 16;
  maxneighs_int2 = 16;
  maxneighs_special = 8;
  kokkos = 1;
  maxatoms = 0;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
int NeighListKokkos<DeviceType>::grow(int nmax)
{
  // skip if this list is already long enough to store nmax atoms
  //  and maxneighs neighbors

  bool return_flag = (nmax <= maxatoms && (int)d_neighbors.extent(1) >= maxneighs);
  if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
    return_flag = (return_flag && d_neighbors_outer.extent(1) >= maxneighs_outer);
  }
  else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
    return_flag = (return_flag && d_neighbors_int2.extent(1) * 2 >= maxneighs_int2);
  }
  if (this->force->pair->use_sep_sepcial) {
    return_flag = (return_flag && d_neighbors_special.extent(1) >= maxneighs_special);    
  }
  if(return_flag) return 0;

  maxatoms = nmax * 1.1;

  MemoryKokkos::realloc_kokkos(k_ilist,"neighlist:ilist",maxatoms);
  d_ilist = k_ilist.view<DeviceType>();
  d_neigh_index = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:neigh_index",maxatoms);
  d_neigh_index_inv = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:neigh_index_inv",maxatoms);
  d_numneigh = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:numneigh",maxatoms);
  d_numfront = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:numfront",maxatoms);
  d_numback = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:numback",maxatoms);
  MemoryKokkos::realloc_kokkos(d_neighbors,"neighlist:neighbors",maxatoms,maxneighs);
  printf("grow: resize d_neighbors to : %d * %d\n", maxatoms, maxneighs);

  if (numneigh_sort) {
    delete [] numneigh_sort;
  }
  numneigh_sort = new int[maxatoms];

  if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
    d_numneigh_outer = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:numneigh_outer",maxatoms);
    MemoryKokkos::realloc_kokkos(d_neighbors_outer,"neighlist:neighbors_outer",maxatoms, maxneighs_outer);
    printf("grow: resize d_numneigh_outer to : %d * %d\n", maxatoms, maxneighs_outer);
  }
  else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
    d_numneigh_int2 = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:numneigh_int2",maxatoms);
    MemoryKokkos::realloc_kokkos(d_neighbors_int2,"neighlist:neighbors_int2",maxatoms, (maxneighs_int2 + 1) / 2);
    printf("grow: resize d_numneigh_int2 to : %d * %d\n", maxatoms, (maxneighs_int2 + 1) / 2);
  }
  
  if (this->force->pair->use_sep_sepcial) {
    printf("grow : resize neighbor special to %d\n", maxneighs_special);
    d_numneigh_special = typename ArrayTypes<DeviceType>::t_int_1d("neighlist:numneigh_special",maxatoms);
    MemoryKokkos::realloc_kokkos(d_neighbors_special,"neighlist:neighbors_special",maxatoms, maxneighs_special);
    printf("grow: resize d_neighbors_special to : %d * %d\n", maxatoms, maxneighs_special);
  }

  if (lmp->kokkos->neigh_transpose) {
    d_neighbors_transpose = typename ArrayTypes<DeviceType>::t_neighbors_2d_lr();
    d_neighbors_transpose = typename ArrayTypes<DeviceType>::t_neighbors_2d_lr(Kokkos::NoInit("neighlist:neighbors"),maxatoms,maxneighs);
  }
  return 1;
}

/* ---------------------------------------------------------------------- */

namespace LAMMPS_NS {
template class NeighListKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class NeighListKokkos<LMPHostType>;
#endif
}

