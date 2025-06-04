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
PairStyle(lj/charmm/coul/long/kk,PairLJCharmmCoulLongKokkos<LMPDeviceType>);
PairStyle(lj/charmm/coul/long/kk/device,PairLJCharmmCoulLongKokkos<LMPDeviceType>);
PairStyle(lj/charmm/coul/long/kk/host,PairLJCharmmCoulLongKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_LJ_CHARMM_COUL_LONG_KOKKOS_H
#define LMP_PAIR_LJ_CHARMM_COUL_LONG_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_lj_charmm_coul_long.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairLJCharmmCoulLongKokkos : public PairLJCharmmCoulLong {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=1};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairLJCharmmCoulLongKokkos(class LAMMPS *);
  ~PairLJCharmmCoulLongKokkos() override;

  void compute(int, int) override;

  void summary() override;

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

  typename AT::t_x_array_randomread bin_base;
  X_FLOAT binsizex, binsizey, binsizez;

  DAT::tdual_efloat_1d k_eatom;
  DAT::tdual_virial_array k_vatom;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  bool ev_allocated = false;
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array;

  bool x_float_allocated = false;
  Kokkos::View<float*[3], Kokkos::LayoutRight, Kokkos::CudaSpace> x_float;

  int x_floatq_size = 0;
  AoS_floatq* x_floatq = nullptr;
  AoS_halfq* x_halfq = nullptr;

  int x_doubleq_size = 0;
  AoS_doubleq* x_doubleq = nullptr;

  // float version of f
  int f_float_size = 0;
  float* f_float = nullptr;
  
  int newton_pair;

  typename AT::tdual_ffloat_2d k_cutsq;
  typename AT::t_ffloat_2d d_cutsq;
  typename AT::t_ffloat_2d d_cut_ljsq;
  typename AT::t_ffloat_2d d_cut_coulsq;

  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split;

  typename AT::t_ffloat_1d_randomread
    d_rtable, d_drtable, d_ftable, d_dftable,
    d_ctable, d_dctable, d_etable, d_detable;

  float *d_rtable_f, *d_drtable_f, *d_ftable_f, *d_dftable_f, 
        *d_ctable_f, *d_dctable_f, *d_etable_f, *d_detable_f;

  __half2* coul_ftable;
  __half2* coul_etable;

  float2* coul_ftable_f;
  float2* coul_etable_f;

  __half2* lj_ftable;

  float2* lj_ftable_f;

  bool param_inited = false;
  bool cuda_var_inited = false;
  __half* coul_ftable_x;
  Coul_LJ_entry* coul_lj_table;

  // map q value to a index
  int* q_val_idx_map;
  int* q_val_idx_mask;
  // q value array indexed by index
  float* q_val_arr;
  // number of q values
  Kokkos::DualView<int*, DeviceType> q_val_num;

  float2* lj_param_table_f;
  float2* lj_param_table_upper_f;
  double2* lj_param_table;
  double2* lj_param_table_upper;

  int neighflag;
  int nlocal,nall,eflag,vflag;

  double special_coul[4];
  double special_lj[4];
  double qqrd2e;

  double cuda_kernel_time;

  void allocate() override;

  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,true,1,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALF,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALFTHREAD,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,false,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,false,1,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALF,false,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,FULL,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,FULL,1,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,HALF,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLJCharmmCoulLongKokkos,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,
                                                            NeighListKokkos<DeviceType>*);
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,true,1,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALF,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALFTHREAD,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,false,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,FULL,false,1,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALF,false,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<0>>;
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,FULL,0,CoulLongTable<0>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,FULL,1,CoulLongTable<0>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,HALF,0,CoulLongTable<0>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmCoulLongKokkos,HALFTHREAD,0,CoulLongTable<0>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLJCharmmCoulLongKokkos,CoulLongTable<0>>(PairLJCharmmCoulLongKokkos*,
                                                            NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairLJCharmmCoulLongKokkos>(PairLJCharmmCoulLongKokkos*);

  friend struct RhodoErfcKernels::PairComputeFunctorCustomDouble<PairLJCharmmCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend struct RhodoErfcKernels::PairComputeFunctorCustomFloat<PairLJCharmmCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT RhodoErfcKernels::pair_compute_neighlist_custom<PairLJCharmmCoulLongKokkos,DOUBLE_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoErfcKernels::pair_compute_custom<PairLJCharmmCoulLongKokkos,DOUBLE_PREC,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);  
  friend EV_FLOAT RhodoErfcKernels::pair_compute_neighlist_custom<PairLJCharmmCoulLongKokkos,FLOAT_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoErfcKernels::pair_compute_custom<PairLJCharmmCoulLongKokkos,FLOAT_PREC,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);  

  friend struct RhodoErfcKernelsExpr::PairComputeFunctorCustomDouble<PairLJCharmmCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend struct RhodoErfcKernelsExpr::PairComputeFunctorCustomFloat<PairLJCharmmCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend struct RhodoErfcKernelsExpr::PairComputeFunctorCustomHfmix<PairLJCharmmCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT RhodoErfcKernelsExpr::pair_compute_neighlist_custom<PairLJCharmmCoulLongKokkos,DOUBLE_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoErfcKernelsExpr::pair_compute_custom<PairLJCharmmCoulLongKokkos,DOUBLE_PREC,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);  
  friend EV_FLOAT RhodoErfcKernelsExpr::pair_compute_neighlist_custom<PairLJCharmmCoulLongKokkos,FLOAT_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoErfcKernelsExpr::pair_compute_custom<PairLJCharmmCoulLongKokkos,FLOAT_PREC,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);  
  friend EV_FLOAT RhodoErfcKernelsExpr::pair_compute_neighlist_custom<PairLJCharmmCoulLongKokkos,HFMIX_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT RhodoErfcKernelsExpr::pair_compute_custom<PairLJCharmmCoulLongKokkos,HFMIX_PREC,CoulLongTable<1>>(PairLJCharmmCoulLongKokkos*,NeighListKokkos<DeviceType>*);  
};

}

#endif
#endif

