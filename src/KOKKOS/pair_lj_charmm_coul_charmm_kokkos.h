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
PairStyle(lj/charmm/coul/charmm/kk,PairLJCharmmCoulCharmmKokkos<LMPDeviceType>);
PairStyle(lj/charmm/coul/charmm/kk/device,PairLJCharmmCoulCharmmKokkos<LMPDeviceType>);
PairStyle(lj/charmm/coul/charmm/kk/host,PairLJCharmmCoulCharmmKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_LJ_CHARMM_COUL_CHARMM_KOKKOS_H
#define LMP_PAIR_LJ_CHARMM_COUL_CHARMM_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_lj_charmm_coul_charmm.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairLJCharmmCoulCharmmKokkos : public PairLJCharmmCoulCharmm {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=1};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairLJCharmmCoulCharmmKokkos(class LAMMPS *);
  ~PairLJCharmmCoulCharmmKokkos() override;

  void compute(int, int) override;

  void settings(int, char **) override;
  void init_tables(double cut_coul, double *cut_respa) override;
  void init_style() override;
  double init_one(int, int) override;


 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int&j,
                        const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fcoul(const F_FLOAT& rsq, const int& i, const int&j, const int& itype,
                        const int& jtype, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int&j,
                        const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_ecoul(const F_FLOAT& rsq, const int& i, const int&j,
                        const int& itype, const int& jtype, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const;

  Kokkos::DualView<params_lj_coul**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_lj_coul**,
    Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  // hardwired to space for 12 atom types
  params_lj_coul m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];

  F_FLOAT m_cutsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  F_FLOAT m_cut_ljsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  F_FLOAT m_cut_coulsq[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];
  typename AT::t_x_array_randomread x;
  typename AT::t_x_array c_x;
  typename AT::t_f_array f;
  typename AT::t_int_1d_randomread type;
  typename AT::t_float_1d_randomread q;

  typename AT::t_x_array_randomread x_rel;
  typename AT::t_x_array c_x_rel;
  typename AT::t_x_array_randomread bin_base;
  X_FLOAT binsizex, binsizey, binsizez;
  half binsizex_h, binsizey_h, binsizez_h;
  typename AT::t_int_1d_randomread fhcut_split;
  half qqrd2e_h, cut_ljsq_h, cut_lj_innersq_h, denom_lj_h, cut_coulsq_h, cut_coul_innersq_h, denom_coul_h;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  bool ev_allocated = false;
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array;

  bool x_float_allocated = false;
  Kokkos::View<float*[3], Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel;

  int x_half_size = 0;
  AoS_half* x_half = nullptr;
  AoS_half* x_half_rel = nullptr;

  int newton_pair;

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;
  typename AT::t_ffloat_2d d_cut_ljsq;
  typename AT::t_ffloat_2d d_cut_coulsq;

  typename AT::t_ffloat_1d_randomread
    d_rtable, d_drtable, d_ftable, d_dftable,
    d_ctable, d_dctable, d_etable, d_detable;

  int neighflag;
  int nlocal,nall,eflag,vflag;

  double special_coul[4];
  double special_lj[4];
  double qqrd2e;

  void allocate() override;

  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,true,1,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALF,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,false,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,false,1,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALF,false,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,FULL,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,FULL,1,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,HALF,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLJCharmmCoulCharmmKokkos,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,
                                                            NeighListKokkos<DeviceType>*);
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,true,1,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALF,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,false,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,FULL,false,1,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALF,false,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,0,CoulLongTable<0>>;
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,FULL,0,CoulLongTable<0>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,FULL,1,CoulLongTable<0>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,HALF,0,CoulLongTable<0>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,0,CoulLongTable<0>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLJCharmmCoulCharmmKokkos,CoulLongTable<0>>(PairLJCharmmCoulCharmmKokkos*,
                                                            NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairLJCharmmCoulCharmmKokkos>(PairLJCharmmCoulCharmmKokkos*);

  friend struct RhodoKernels::PairComputeFunctorCustomDouble<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,0,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomDouble<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,1,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomFloat<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,0,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomFloat<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,1,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomHalf<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,0,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomHalf<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,1,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomFhmix<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,0,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomFhmix<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,1,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomDfmix<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,0,0,CoulLongTable<1>>;
  friend struct RhodoKernels::PairComputeFunctorCustomDfmix<PairLJCharmmCoulCharmmKokkos,HALFTHREAD,false,1,0,CoulLongTable<1>>;
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,DOUBLE_PREC,HALFTHREAD,0,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,DOUBLE_PREC,HALFTHREAD,1,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_custom<PairLJCharmmCoulCharmmKokkos,DOUBLE_PREC,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);  
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,FLOAT_PREC,HALFTHREAD,0,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,FLOAT_PREC,HALFTHREAD,1,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_custom<PairLJCharmmCoulCharmmKokkos,FLOAT_PREC,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);  
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,HALF_PREC,HALFTHREAD,0,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,HALF_PREC,HALFTHREAD,1,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_custom<PairLJCharmmCoulCharmmKokkos,HALF_PREC,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);  
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,HFMIX_PREC,HALFTHREAD,0,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,HFMIX_PREC,HALFTHREAD,1,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_custom<PairLJCharmmCoulCharmmKokkos,HFMIX_PREC,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);  
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,DFMIX_PREC,HALFTHREAD,0,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_neighlist_custom<PairLJCharmmCoulCharmmKokkos,DFMIX_PREC,HALFTHREAD,1,0,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoKernels::pair_compute_custom<PairLJCharmmCoulCharmmKokkos,DFMIX_PREC,CoulLongTable<1>>(PairLJCharmmCoulCharmmKokkos*,NeighListKokkos<DeviceType>*);  
};

}

#endif
#endif

