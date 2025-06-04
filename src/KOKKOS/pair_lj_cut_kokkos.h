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

#ifdef PAIR_CLASS
// clang-format off
PairStyle(lj/cut/kk,PairLJCutKokkos<LMPDeviceType>);
PairStyle(lj/cut/kk/device,PairLJCutKokkos<LMPDeviceType>);
PairStyle(lj/cut/kk/host,PairLJCutKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_LJ_CUT_KOKKOS_H
#define LMP_PAIR_LJ_CUT_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_lj_cut.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairLJCutKokkos : public PairLJCut {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=0};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairLJCutKokkos(class LAMMPS *);
  ~PairLJCutKokkos() override;

  void compute(int, int) override;

  void summary() override;

  void settings(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;

  struct params_lj{
    KOKKOS_INLINE_FUNCTION
    params_lj() {cutsq=0,lj1=0;lj2=0;lj3=0;lj4=0;offset=0;};
    KOKKOS_INLINE_FUNCTION
    params_lj(int /*i*/) {cutsq=0,lj1=0;lj2=0;lj3=0;lj4=0;offset=0;};
    F_FLOAT cutsq,lj1,lj2,lj3,lj4,offset;
  };

 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_ecoul(const F_FLOAT& /*rsq*/, const int& /*i*/, const int& /*j*/,
                        const int& /*itype*/, const int& /*jtype*/) const { return 0; }

  Kokkos::DualView<params_lj**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_lj**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_lj m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];  // hardwired to space for 12 atom types
  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename AT::t_x_array_randomread x;
  typename AT::t_x_array c_x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;

  typename AT::t_x_array_randomread x_rel;
  typename AT::t_x_array c_x_rel;
  typename AT::t_x_array_randomread bin_base;
  X_FLOAT binsizex, binsizey, binsizez;
  half binsizex_h, binsizey_h, binsizez_h;
  typename AT::t_int_1d_randomread fhcut_split;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  bool ev_allocated = false;
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array;

  bool x_float_allocated = false;
  Kokkos::View<float*[3], Kokkos::LayoutRight, Kokkos::CudaSpace> x_float;
  Kokkos::View<float*[3], Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel;

  int x_half_size = 0;
  AoS_half* x_half = nullptr;
  AoS_half* x_half_rel = nullptr;
  AoS_half_xonly* x_half_xonly = nullptr;
  AoS_half_xonly* x_half_rel_xonly = nullptr;

  int newton_pair;
  double special_lj[4];

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;


  int neighflag;
  int nlocal,nall,eflag,vflag;

  double pair_compute_time;
  double cuda_kernel_time;
  double init_time;

  void allocate() override;
  friend struct PairComputeFunctor<PairLJCutKokkos,FULL,true,0>;
  friend struct PairComputeFunctor<PairLJCutKokkos,FULL,true,1>;
  friend struct PairComputeFunctor<PairLJCutKokkos,HALF,true>;
  friend struct PairComputeFunctor<PairLJCutKokkos,HALFTHREAD,true>;
  friend struct PairComputeFunctor<PairLJCutKokkos,FULL,false,0>;
  friend struct PairComputeFunctor<PairLJCutKokkos,FULL,false,1>;
  friend struct PairComputeFunctor<PairLJCutKokkos,HALF,false>;
  friend struct PairComputeFunctor<PairLJCutKokkos,HALFTHREAD,false>;
  friend EV_FLOAT pair_compute_neighlist<PairLJCutKokkos,FULL,0>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCutKokkos,FULL,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCutKokkos,HALF>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCutKokkos,HALFTHREAD>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLJCutKokkos>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairLJCutKokkos>(PairLJCutKokkos*);

  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernels::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;

  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernels::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;

  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernels::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;

  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernels::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;


  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomDouble<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;

  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFloat<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;

  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomFhmix<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;

  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,NO_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,BASIC_NEIGH_SEP,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,0,TWO_END_NEIGH_INT2,1>;
  friend struct LJKernelsExpr::PairComputeFunctorCustomHalf<PairLJCutKokkos,FULL,true,1,TWO_END_NEIGH_INT2,1>;

  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_custom<PairLJCutKokkos,DOUBLE_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_custom<PairLJCutKokkos,FLOAT_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_custom<PairLJCutKokkos,HALF_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernels::pair_compute_custom<PairLJCutKokkos,HFMIX_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);

  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,DOUBLE_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_custom<PairLJCutKokkos,DOUBLE_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,FLOAT_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_custom<PairLJCutKokkos,FLOAT_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HFMIX_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_custom<PairLJCutKokkos,HFMIX_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,NO_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,BASIC_NEIGH_SEP,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,TWO_END_NEIGH,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,0,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_neighlist_custom<PairLJCutKokkos,HALF_PREC,FULL,1,TWO_END_NEIGH_INT2,1>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT LJKernelsExpr::pair_compute_custom<PairLJCutKokkos,HALF_PREC>(PairLJCutKokkos*,NeighListKokkos<DeviceType>*);

  // friend void double_force_kernel<PairLJCutKokkos>(int ntotal, const NeighListKokkos<DeviceType> list, 
  //   PairLJCutKokkos c, typename ArrayTypes<DeviceType>::t_f_array f);
  //friend void test_kernel<PairLJCutKokkos>(int ntotal, const NeighListKokkos<DeviceType> list, 
  //  PairLJCutKokkos c, typename ArrayTypes<DeviceType>::t_f_array f, double* f_ptr);
};

}

#endif
#endif

