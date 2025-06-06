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
PairStyle(lj/charmmfsw/coul/long/kk,PairLJCharmmfswCoulLongKokkos<LMPDeviceType>);
PairStyle(lj/charmmfsw/coul/long/kk/device,PairLJCharmmfswCoulLongKokkos<LMPDeviceType>);
PairStyle(lj/charmmfsw/coul/long/kk/host,PairLJCharmmfswCoulLongKokkos<LMPHostType>);
// clang-format on
#else

// clang-format off
#ifndef LMP_PAIR_LJ_CHARMMFSW_COUL_LONG_KOKKOS_H
#define LMP_PAIR_LJ_CHARMMFSW_COUL_LONG_KOKKOS_H

#include "pair_kokkos.h"
#include "pair_lj_charmmfsw_coul_long.h"
#include "neigh_list_kokkos.h"

namespace LAMMPS_NS {

template<class DeviceType>
class PairLJCharmmfswCoulLongKokkos : public PairLJCharmmfswCoulLong {
 public:
  enum {EnabledNeighFlags=FULL|HALFTHREAD|HALF};
  enum {COUL_FLAG=1};
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  PairLJCharmmfswCoulLongKokkos(class LAMMPS *);
  ~PairLJCharmmfswCoulLongKokkos() override;

  void compute(int, int) override;

  void init_tables(double cut_coul, double *cut_respa) override;
  void init_style() override;
  double init_one(int, int) override;

 protected:
  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fpair(const F_FLOAT& rsq, const int& i, const int& j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_evdwl(const F_FLOAT& rsq, const int& i, const int& j, const int& itype, const int& jtype) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_fcoul(const F_FLOAT& rsq, const int& i, const int& j, const int& itype,
                        const int& jtype, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const;

  template<bool STACKPARAMS, class Specialisation>
  KOKKOS_INLINE_FUNCTION
  F_FLOAT compute_ecoul(const F_FLOAT& rsq, const int& i, const int& j, const int& itype,
                        const int& jtype, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const;

  Kokkos::DualView<params_lj_coul**,Kokkos::LayoutRight,DeviceType> k_params;
  typename Kokkos::DualView<params_lj_coul**,Kokkos::LayoutRight,DeviceType>::t_dev_const_um params;
  params_lj_coul m_params[MAX_TYPES_STACKPARAMS+1][MAX_TYPES_STACKPARAMS+1];  // hardwired to space for 12 atom types
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

  float* coul_ftable_f;

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

  float2 *param_lj12_f, *param_lj34_f;
  double2 *param_lj12, *param_lj34;

  int rsq_array_size = 0;
  float *max_inner_rsq, *min_outer_rsq; 

  bool param_inited = false;
  bool q_val_inited = false;
  // map q value to a index
  int* q_val_idx_map;
  int* q_val_idx_mask;
  // q value array indexed by index
  float* q_val_arr;
  // number of q values
  Kokkos::DualView<int*, DeviceType> q_val_num;

  int neighflag;
  int nlocal,nall,eflag,vflag;

  double special_lj[4];
  double special_coul[4];
  double qqrd2e;

  void allocate() override;

  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,true,1,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALF,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,true,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,false,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,false,1,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALF,false,0,CoulLongTable<1>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,FULL,0,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,FULL,1,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,HALF,0,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLJCharmmfswCoulLongKokkos,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,
                                                            NeighListKokkos<DeviceType>*);
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,true,1,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALF,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,true,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,false,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,FULL,false,1,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALF,false,0,CoulLongTable<0>>;
  friend struct PairComputeFunctor<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<0>>;
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,FULL,0,CoulLongTable<0>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,FULL,1,CoulLongTable<0>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,HALF,0,CoulLongTable<0>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute_neighlist<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,0,CoulLongTable<0>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT pair_compute<PairLJCharmmfswCoulLongKokkos,CoulLongTable<0>>(PairLJCharmmfswCoulLongKokkos*,
                                                            NeighListKokkos<DeviceType>*);
  friend void pair_virial_fdotr_compute<PairLJCharmmfswCoulLongKokkos>(PairLJCharmmfswCoulLongKokkos*);

  friend struct CharmmfswCoulLongKernels::PairComputeFunctorCustomDouble<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT CharmmfswCoulLongKernels::pair_compute_neighlist_custom<PairLJCharmmfswCoulLongKokkos,DOUBLE_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT CharmmfswCoulLongKernels::pair_compute_custom<PairLJCharmmfswCoulLongKokkos,DOUBLE_PREC,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);  
  friend struct CharmmfswCoulLongKernels::PairComputeFunctorCustomFloat<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT CharmmfswCoulLongKernels::pair_compute_neighlist_custom<PairLJCharmmfswCoulLongKokkos,FLOAT_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT CharmmfswCoulLongKernels::pair_compute_custom<PairLJCharmmfswCoulLongKokkos,FLOAT_PREC,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);  
  friend struct CharmmfswCoulLongKernels::PairComputeFunctorCustomHfmix<PairLJCharmmfswCoulLongKokkos,HALFTHREAD,false,0,CoulLongTable<1>>;
  friend EV_FLOAT CharmmfswCoulLongKernels::pair_compute_neighlist_custom<PairLJCharmmfswCoulLongKokkos,HFMIX_PREC,HALFTHREAD,0,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);
  friend EV_FLOAT CharmmfswCoulLongKernels::pair_compute_custom<PairLJCharmmfswCoulLongKokkos,HFMIX_PREC,CoulLongTable<1>>(PairLJCharmmfswCoulLongKokkos*,NeighListKokkos<DeviceType>*);  
};

}

#endif
#endif

