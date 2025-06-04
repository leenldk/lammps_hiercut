// clang-format off
/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifndef LMP_NEIGH_LIST_KOKKOS_H
#define LMP_NEIGH_LIST_KOKKOS_H

#include "pointers.h"
#include "force.h"

#include "neigh_list.h"         // IWYU pragma: export
#include "kokkos_type.h"

namespace LAMMPS_NS {

class AtomNeighbors
{
 public:
  int num_neighs;

  KOKKOS_INLINE_FUNCTION
  AtomNeighbors(): num_neighs(0), _firstneigh(NULL), _stride(0) {}
  KOKKOS_INLINE_FUNCTION
  AtomNeighbors(int* const & firstneigh, const int & _num_neighs,
                const int & stride):
  num_neighs(_num_neighs), _firstneigh(firstneigh), _stride(stride) {};
  KOKKOS_INLINE_FUNCTION
  int& operator()(const int &i) const {
    return _firstneigh[(bigint) i*_stride];
  }

// private:
  int* _firstneigh;
  int _stride;
};

class AtomNeighbors_int2
{
 public:
  int num_neighs;
  int2* _firstneigh;
  int _stride;

  KOKKOS_INLINE_FUNCTION
  AtomNeighbors_int2(): num_neighs(0), _firstneigh(NULL), _stride(0) {}
  KOKKOS_INLINE_FUNCTION
  AtomNeighbors_int2(int2* const & firstneigh, const int & _num_neighs,
                const int & stride):
  num_neighs(_num_neighs), _firstneigh(firstneigh), _stride(stride) {};
  KOKKOS_INLINE_FUNCTION
  int2& operator()(const int &i) const {
    return _firstneigh[(bigint) i*_stride];
  }

};


class AtomNeighborsConst
{
 public:
  const int* const _firstneigh;
  const int num_neighs;

  KOKKOS_INLINE_FUNCTION
  AtomNeighborsConst(const int* const & firstneigh, const int & _num_neighs,
                     const int & stride):
  _firstneigh(firstneigh), num_neighs(_num_neighs), _stride(stride) {};
  KOKKOS_INLINE_FUNCTION
  const int& operator()(const int &i) const {
    return _firstneigh[(bigint) i*_stride];
  }

// private:
  //const int* const _firstneigh;
  const int _stride;
};

class AtomNeighborsConst_int2
{
 public:
  const int2* const _firstneigh;
  const int num_neighs;

  KOKKOS_INLINE_FUNCTION
  AtomNeighborsConst_int2(const int2* const & firstneigh, const int & _num_neighs,
                     const int & stride):
  _firstneigh(firstneigh), num_neighs(_num_neighs), _stride(stride) {};
  KOKKOS_INLINE_FUNCTION
  const int2& operator()(const int &i) const {
    return _firstneigh[(bigint) i*_stride];
  }

// private:
  //const int* const _firstneigh;
  const int _stride;
};

template<class DeviceType>
class NeighListKokkos: public NeighList {
  int _stride;

public:
  int maxneighs;
  // outer layer atoms neighbor num in MULTI_NEIGH_LIST strategy
  int maxneighs_outer;
  int maxneighs_int2;
  int maxneighs_special;
  int maxatoms;

  // return 1 if actually grow, 0 otherwise
  int grow(int nmax);
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors;
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_outer;
  typename ArrayTypes<DeviceType>::t_neighbors_2d_int2 d_neighbors_int2;
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special;
  typename ArrayTypes<DeviceType>::t_neighbors_2d_lr d_neighbors_transpose;
  DAT::tdual_int_1d k_ilist;   // local indices of I atoms
  typename ArrayTypes<DeviceType>::t_int_1d d_ilist;
  typename ArrayTypes<DeviceType>::t_int_1d d_neigh_index;
  typename ArrayTypes<DeviceType>::t_int_1d d_neigh_index_inv;
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh;
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_outer;
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_int2;
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special;
  typename ArrayTypes<DeviceType>::t_int_1d d_numfront;
  typename ArrayTypes<DeviceType>::t_int_1d d_numback;

  NeighListKokkos(class LAMMPS *lmp);

  KOKKOS_INLINE_FUNCTION
  AtomNeighbors get_neighbors(const int &i) const {
    return AtomNeighbors(&d_neighbors(i,0),d_numneigh(i),
                         &d_neighbors(i,1)-&d_neighbors(i,0));
  }

  KOKKOS_INLINE_FUNCTION
  AtomNeighbors get_neighbors_special(const int &i) const {
    return AtomNeighbors(&d_neighbors_special(i,0),d_numneigh_special(i),
                         &d_neighbors_special(i,1)-&d_neighbors_special(i,0));
  }

  KOKKOS_INLINE_FUNCTION
  AtomNeighbors get_neighbors_outer(const int &i) const {
    return AtomNeighbors(&d_neighbors_outer(i,0),d_numneigh_outer(i),
                         &d_neighbors_outer(i,1)-&d_neighbors_outer(i,0));
  }

  KOKKOS_INLINE_FUNCTION
  AtomNeighbors_int2 get_neighbors_int2(const int &i) const {
    return AtomNeighbors_int2(&d_neighbors_int2(i,0),d_numneigh_int2(i),
                         &d_neighbors_int2(i,1)-&d_neighbors_int2(i,0));
  }

  KOKKOS_INLINE_FUNCTION
  AtomNeighbors get_neighbors_transpose(const int &i) const {
    return AtomNeighbors(&d_neighbors_transpose(i,0),d_numneigh(i),
                         &d_neighbors_transpose(i,1)-&d_neighbors_transpose(i,0));
  }

  KOKKOS_INLINE_FUNCTION
  static AtomNeighborsConst static_neighbors_const(int i,
           typename ArrayTypes<DeviceType>::t_neighbors_2d_const const& d_neighbors,
           typename ArrayTypes<DeviceType>::t_int_1d_const const& d_numneigh) {
    return AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),
                              &d_neighbors(i,1)-&d_neighbors(i,0));
  }

  KOKKOS_INLINE_FUNCTION
  AtomNeighborsConst get_neighbors_const(const int &i) const {
    return AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),
                              &d_neighbors(i,1)-&d_neighbors(i,0));
  }

  KOKKOS_INLINE_FUNCTION
  int& num_neighs(const int & i) const {
    return d_numneigh(i);
  }
};

}

#endif
