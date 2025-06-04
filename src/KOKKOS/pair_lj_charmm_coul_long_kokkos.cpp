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

/* ----------------------------------------------------------------------
   Contributing author: Ray Shan (SNL)
------------------------------------------------------------------------- */

#include "pair_lj_charmm_coul_long_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "ewald_const.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

#include <cmath>
#include <cstring>

using namespace LAMMPS_NS;
using namespace EwaldConst;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCharmmCoulLongKokkos<DeviceType>::PairLJCharmmCoulLongKokkos(LAMMPS *lmp):PairLJCharmmCoulLong(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | Q_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  
  cuda_kernel_time = 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCharmmCoulLongKokkos<DeviceType>::~PairLJCharmmCoulLongKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }
}

/* ---------------------------------------------------------------------- */

namespace InitKernel {

typedef union {
  int i;
  float f;
} union_int_float_t;

template<class DeviceType>
__global__ void init_coul_table(int table_size, __half2 *coul_ftable, __half2 *coul_etable, float2 *coul_ftable_f, float2 *coul_etable_f, int ncoulmask, int ncoulshiftbits,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < table_size) {
    __half rsq_h = __ushort_as_half(static_cast<ushort>(i));
    float rsq_f = __half2float(rsq_h);
    double rsq_d = (double)rsq_f;

    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq_f;
    const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
    {
      const double fraction = (rsq_d - d_rtable[itable]) * d_drtable[itable];
      const double table = d_ftable[itable] + fraction * d_dftable[itable];
      const float f1_rsq_f = table / rsq_d;
      const __half f1_rsq = __float2half(f1_rsq_f);

      const double table1 = d_ctable[itable] + fraction * d_dctable[itable];
      const float f2_rsq_f = table1 / rsq_d;
      const __half f2_rsq = __float2half(f2_rsq_f);

      coul_ftable[i] = __halves2half2(f1_rsq, f2_rsq);
      coul_ftable_f[i] = make_float2(f1_rsq_f, f2_rsq_f);
    }
    {
      const double fraction = (rsq_d - d_rtable[itable]) * d_drtable[itable];
      const double table = d_etable[itable] + fraction * d_detable[itable];
      const float f1_rsq_f = table;
      const __half f1_rsq = __double2half(table);

      const double table1 = d_ctable[itable] + fraction * d_dctable[itable];
      const float f2_rsq_f = table1;
      const __half f2_rsq = __double2half(table1);

      coul_etable[i] = __halves2half2(f1_rsq, f2_rsq);
      coul_etable_f[i] = make_float2(f1_rsq_f, f2_rsq_f);
    }
  }
}

template<class DeviceType>
__global__ void init_lj_table(int table_size, __half2 *lj_ftable, float2 *lj_ftable_f,
  double cut_ljsq, double cut_lj_innersq, double denom_lj)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < table_size) {
    __half rsq_h = __ushort_as_half(static_cast<ushort>(i));
    float rsq_f = __half2float(rsq_h);
    double rsq_d = (double)rsq_f;

    if (rsq_d > cut_lj_innersq) {
      double switch1 = (cut_ljsq-rsq_d) * (cut_ljsq-rsq_d) *
                (cut_ljsq + 2.0*rsq_d - 3.0*cut_lj_innersq) / denom_lj;
      double switch2 = 12.0*rsq_d * (cut_ljsq-rsq_d) * (rsq_d-cut_lj_innersq) / denom_lj;
      double r2inv = 1.0 / rsq_d;
      double r6inv = r2inv * r2inv * r2inv;
      float f1_rsq = (float)(switch1);
      float f2_rsq = (float)(r6inv * switch2);

      lj_ftable[i] = __halves2half2(__float2half(f1_rsq), __float2half(f2_rsq));
      lj_ftable_f[i] = make_float2(f1_rsq, f2_rsq);
    }
    else {
      lj_ftable[i] = __halves2half2(__float2half(1.0f), __float2half(0.0f));
      lj_ftable_f[i] = make_float2(1.0f, 0.0f);
    }
  }  
}

template<class DeviceType>
__global__ void init_lj_param_table(int dim_size, float2 *lj_param_table_f, double2 *lj_param_table,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  int col = blockIdx.y * blockDim.y + threadIdx.y;
  if (row < dim_size && col < dim_size) {
    float2 f;
    f.x = params(row, col).lj3;
    f.y = params(row, col).lj4;
    lj_param_table_f[row * dim_size + col] = f;
    double2 d;
    d.x = params(row, col).lj3;
    d.y = params(row, col).lj4;
    lj_param_table[row * dim_size + col] = d;
    //printf("set lj_param_table_f %d %d, %d to %f %f\n", row, col, row * dim_size + col, f.x, f.y);
  }
}

template<class DeviceType>
__global__ void init_lj_param_table_upper(int dim_size, float2 *lj_param_table_f, double2 *lj_param_table,
  float2 *lj_param_table_upper_f, double2 *lj_param_table_upper)
{
  int prefix_sum = 0;
  for(int i = 0; i < threadIdx.x; i++) {
    prefix_sum += dim_size - i;
  }
  for(int i = 0; i < dim_size - threadIdx.x; i++) {
    lj_param_table_upper_f[prefix_sum + i] = lj_param_table_f[(threadIdx.x + 1) * (dim_size + 1) + (threadIdx.x + i + 1)];
    lj_param_table_upper[prefix_sum + i] = lj_param_table[(threadIdx.x + 1) * (dim_size + 1) + (threadIdx.x + i + 1)];
  }
}

template<class DeviceType>
__global__ void copy_d_table(int table_size, float* d_rtable_f, float* d_drtable_f, float* d_ftable_f, float* d_dftable_f,
  float* d_ctable_f, float* d_dctable_f, float* d_etable_f, float* d_detable_f,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < table_size) {
    d_rtable_f[i] = d_rtable[i];
    d_drtable_f[i] = d_drtable[i];
    d_ftable_f[i] = d_ftable[i];
    d_dftable_f[i] = d_dftable[i];
    d_ctable_f[i] = d_ctable[i];
    d_dctable_f[i] = d_dctable[i];
    d_etable_f[i] = d_etable[i];
    d_detable_f[i] = d_detable[i];
  }
}

}

template<class DeviceType>
void PairLJCharmmCoulLongKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  eflag = eflag_in;
  vflag = vflag_in;

  if (neighflag == FULL) no_virial_fdotr_compute = 1;

  ev_init(eflag,vflag,0);

  // reallocate per-atom arrays if necessary

  if (eflag_atom) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->create_kokkos(k_eatom,eatom,maxeatom,"pair:eatom");
    d_eatom = k_eatom.view<DeviceType>();
  }
  if (vflag_atom) {
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->create_kokkos(k_vatom,vatom,maxvatom,"pair:vatom");
    d_vatom = k_vatom.view<DeviceType>();
  }

  atomKK->sync(execution_space,datamask_read);
  k_cutsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();
  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];
  special_coul[0] = force->special_coul[0];
  special_coul[1] = force->special_coul[1];
  special_coul[2] = force->special_coul[2];
  special_coul[3] = force->special_coul[3];
  qqrd2e = force->qqrd2e;
  newton_pair = force->newton_pair;

  bin_base = atomKK->k_bin_base.view<DeviceType>();
  binsizex = atomKK->binsizex;
  binsizey = atomKK->binsizey;
  binsizez = atomKK->binsizez;  
  fhcut_split = atomKK->k_fhcut_split.view<DeviceType>();

  // loop over neighbors of my atoms

  if(!param_inited) {
    param_inited = true;
    {
      int table_size = 1<<16;
      cudaMalloc((void**)&coul_ftable, table_size * sizeof(__half2));
      cudaMalloc((void**)&coul_etable, table_size * sizeof(__half2));
      cudaMalloc((void**)&coul_ftable_f, table_size * sizeof(float2));
      cudaMalloc((void**)&coul_etable_f, table_size * sizeof(float2));
      
      int threadsPerBlock = 128;
      int blocksPerGrid = (table_size + threadsPerBlock - 1) / threadsPerBlock;
      InitKernel::init_coul_table<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(
        table_size, coul_ftable, coul_etable, coul_ftable_f, coul_etable_f, ncoulmask, ncoulshiftbits,
        d_rtable, d_drtable, d_ftable, d_dftable, d_ctable, d_dctable, d_etable, d_detable);
    }
    {  
      int table_size = 1<<16;
      cudaMalloc((void**)&lj_ftable, table_size * sizeof(__half2));
      cudaMalloc((void**)&lj_ftable_f, table_size * sizeof(float2));

      int threadsPerBlock = 128;
      int blocksPerGrid = (table_size + threadsPerBlock - 1) / threadsPerBlock;
      InitKernel::init_lj_table<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(
        table_size, lj_ftable, lj_ftable_f, cut_ljsq, cut_lj_innersq, denom_lj);
    }
    {
      int dim_size = atom->ntypes + 1;
      printf("init lj param table, dim size : %d\n", dim_size);
      cudaMalloc((void**)&lj_param_table_f, dim_size * dim_size * sizeof(float2));
      cudaMalloc((void**)&lj_param_table_upper_f, dim_size * dim_size * sizeof(float2));
      cudaMalloc((void**)&lj_param_table, dim_size * dim_size * sizeof(double2));
      cudaMalloc((void**)&lj_param_table_upper, dim_size * dim_size * sizeof(double2));

      int block_threads = 8;
      dim3 threadsPerBlock(block_threads, block_threads);
      dim3 blocksPerGrid((dim_size + block_threads - 1) / block_threads, (dim_size + block_threads - 1) / block_threads);
      InitKernel::init_lj_param_table<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(
        dim_size, lj_param_table_f, lj_param_table, params); 

      InitKernel::init_lj_param_table_upper<DeviceType><<<1, atom->ntypes>>>(
        atom->ntypes, lj_param_table_f, lj_param_table, lj_param_table_upper_f, lj_param_table_upper); 
    }
    {
      int table_size = d_rtable.extent(0);
      printf("init float d_table, table size : %d\n", table_size);

      cudaMalloc((void**)&d_rtable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_drtable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_ftable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_dftable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_ctable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_dctable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_etable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_detable_f, table_size * sizeof(float));

      int threadsPerBlock = 128;
      int blocksPerGrid = (table_size + threadsPerBlock - 1) / threadsPerBlock;
      InitKernel::copy_d_table<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(
        table_size, d_rtable_f, d_drtable_f, d_ftable_f, d_dftable_f, d_ctable_f, d_dctable_f, d_etable_f, d_detable_f,
        d_rtable, d_drtable, d_ftable, d_dftable, d_ctable, d_dctable, d_etable, d_detable);
    }
    cudaDeviceSynchronize();
  }

  copymode = 1;

  EV_FLOAT ev;
  // ncoultablebits != 0
  if (this->prec_type == DEFAULT_PREC) {
    // use default precsion
    if (ncoultablebits)
      ev = pair_compute<PairLJCharmmCoulLongKokkos<DeviceType>,CoulLongTable<1> >
        (this,(NeighListKokkos<DeviceType>*)list);
    else
      ev = pair_compute<PairLJCharmmCoulLongKokkos<DeviceType>,CoulLongTable<0> >
        (this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == DOUBLE_PREC) {
    ev = RhodoErfcKernelsExpr::pair_compute_custom<PairLJCharmmCoulLongKokkos<DeviceType>, DOUBLE_PREC, CoulLongTable<1> >(this,(NeighListKokkos<DeviceType>*)list);
    // ev = RhodoErfcKernels::pair_compute_custom<PairLJCharmmCoulLongKokkos<DeviceType>, DOUBLE_PREC, CoulLongTable<1> >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == FLOAT_PREC) {
    ev = RhodoErfcKernelsExpr::pair_compute_custom<PairLJCharmmCoulLongKokkos<DeviceType>, FLOAT_PREC, CoulLongTable<1> >(this,(NeighListKokkos<DeviceType>*)list);
    // ev = RhodoErfcKernels::pair_compute_custom<PairLJCharmmCoulLongKokkos<DeviceType>, FLOAT_PREC, CoulLongTable<1> >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == HFMIX_PREC) {
    ev = RhodoErfcKernelsExpr::pair_compute_custom<PairLJCharmmCoulLongKokkos<DeviceType>, HFMIX_PREC, CoulLongTable<1> >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else {
    // other precision not supported
    error->all(FLERR,"Invalid precision type");
  }

  Kokkos::fence();

  if (eflag) {
    eng_vdwl += ev.evdwl;
    eng_coul += ev.ecoul;
  }
  if (vflag_global) {
    virial[0] += ev.v[0];
    virial[1] += ev.v[1];
    virial[2] += ev.v[2];
    virial[3] += ev.v[3];
    virial[4] += ev.v[4];
    virial[5] += ev.v[5];
  }

  if (eflag_atom) {
    k_eatom.template modify<DeviceType>();
    k_eatom.template sync<LMPHostType>();
  }

  if (vflag_atom) {
    k_vatom.template modify<DeviceType>();
    k_vatom.template sync<LMPHostType>();
  }

  if (vflag_fdotr) pair_virial_fdotr_compute(this);

  copymode = 0;
}

template<class DeviceType>
void PairLJCharmmCoulLongKokkos<DeviceType>::summary()
{
  printf("PairLJCharmmCoulLong::summary\n");
  printf("PairLJCutKokkos::cuda_kernel_time = %lf\n", cuda_kernel_time);
}

/* ----------------------------------------------------------------------
   compute LJ CHARMM pair force between atoms i and j
   ---------------------------------------------------------------------- */
template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCharmmCoulLongKokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& /*i*/, const int& /*j*/,
              const int& itype, const int& jtype) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT r6inv = r2inv*r2inv*r2inv;
  F_FLOAT forcelj, switch1, switch2, englj;

  forcelj = r6inv *
    ((STACKPARAMS?m_params[itype][jtype].lj1:params(itype,jtype).lj1)*r6inv -
     (STACKPARAMS?m_params[itype][jtype].lj2:params(itype,jtype).lj2));

  if (rsq > cut_lj_innersq) {
    switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
              (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
    switch2 = 12.0*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
    englj = r6inv *
            ((STACKPARAMS?m_params[itype][jtype].lj3:params(itype,jtype).lj3)*r6inv -
             (STACKPARAMS?m_params[itype][jtype].lj4:params(itype,jtype).lj4));
    forcelj = forcelj*switch1 + englj*switch2;
  }

  return forcelj*r2inv;
}

/* ----------------------------------------------------------------------
   compute LJ CHARMM pair potential energy between atoms i and j
   ---------------------------------------------------------------------- */
template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCharmmCoulLongKokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& /*i*/, const int& /*j*/,
              const int& itype, const int& jtype) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT r6inv = r2inv*r2inv*r2inv;
  F_FLOAT englj, switch1;

  englj = r6inv *
    ((STACKPARAMS?m_params[itype][jtype].lj3:params(itype,jtype).lj3)*r6inv -
     (STACKPARAMS?m_params[itype][jtype].lj4:params(itype,jtype).lj4));

  if (rsq > cut_lj_innersq) {
    switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
    englj *= switch1;
  }
  return englj;
}

/* ----------------------------------------------------------------------
   compute coulomb pair force between atoms i and j
   ---------------------------------------------------------------------- */
template<class DeviceType>
template<bool STACKPARAMS,  class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCharmmCoulLongKokkos<DeviceType>::
compute_fcoul(const F_FLOAT& rsq, const int& /*i*/, const int&j,
              const int& /*itype*/, const int& /*jtype*/,
              const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const {
  if (Specialisation::DoTable && rsq > tabinnersq) {
    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq;
    const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
    const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
    const F_FLOAT table = d_ftable[itable] + fraction*d_dftable[itable];
    F_FLOAT forcecoul = qtmp*q[j] * table;
    if (factor_coul < 1.0) {
      const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
      const F_FLOAT prefactor = qtmp*q[j] * table;
      forcecoul -= (1.0-factor_coul)*prefactor;
    }
    return forcecoul/rsq;
  } else {
    const F_FLOAT r = sqrt(rsq);
    const F_FLOAT grij = g_ewald * r;
    const F_FLOAT expm2 = exp(-grij*grij);
    const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
    const F_FLOAT rinv = 1.0/r;
    const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
    const F_FLOAT prefactor = qqrd2e * qtmp*q[j]*rinv;
    F_FLOAT forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
    if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;

    return forcecoul*rinv*rinv;
  }
}

/* ----------------------------------------------------------------------
   compute coulomb pair potential energy between atoms i and j
   ---------------------------------------------------------------------- */
template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCharmmCoulLongKokkos<DeviceType>::
compute_ecoul(const F_FLOAT& rsq, const int& /*i*/, const int&j,
              const int& /*itype*/, const int& /*jtype*/, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const {
  if (Specialisation::DoTable && rsq > tabinnersq) {
    union_int_float_t rsq_lookup;
    rsq_lookup.f = rsq;
    const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
    const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
    const F_FLOAT table = d_etable[itable] + fraction*d_detable[itable];
    F_FLOAT ecoul = qtmp*q[j] * table;
    if (factor_coul < 1.0) {
      const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
      const F_FLOAT prefactor = qtmp*q[j] * table;
      ecoul -= (1.0-factor_coul)*prefactor;
    }
    return ecoul;
  } else {
    const F_FLOAT r = sqrt(rsq);
    const F_FLOAT grij = g_ewald * r;
    const F_FLOAT expm2 = exp(-grij*grij);
    const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
    const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
    const F_FLOAT prefactor = qqrd2e * qtmp*q[j]/r;
    F_FLOAT ecoul = prefactor * erfc;
    if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
    return ecoul;
  }
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCharmmCoulLongKokkos<DeviceType>::allocate()
{
  PairLJCharmmCoulLong::allocate();

  int n = atom->ntypes;

  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  d_cut_ljsq = typename AT::t_ffloat_2d("pair:cut_ljsq",n+1,n+1);

  d_cut_coulsq = typename AT::t_ffloat_2d("pair:cut_coulsq",n+1,n+1);

  k_params = Kokkos::DualView<params_lj_coul**,Kokkos::LayoutRight,DeviceType>("PairLJCharmmCoulLong::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

template<class DeviceType>
void PairLJCharmmCoulLongKokkos<DeviceType>::init_tables(double cut_coul, double *cut_respa)
{
  Pair::init_tables(cut_coul,cut_respa);

  typedef typename ArrayTypes<DeviceType>::t_ffloat_1d table_type;
  typedef typename ArrayTypes<LMPHostType>::t_ffloat_1d host_table_type;

  int ntable = 1;
  for (int i = 0; i < ncoultablebits; i++) ntable *= 2;


  // Copy rtable and drtable
  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);
  for (int i = 0; i < ntable; i++) {
    h_table(i) = rtable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_rtable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);
  for (int i = 0; i < ntable; i++) {
    h_table(i) = drtable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_drtable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  // Copy ftable and dftable
  for (int i = 0; i < ntable; i++) {
    h_table(i) = ftable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_ftable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  for (int i = 0; i < ntable; i++) {
    h_table(i) = dftable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_dftable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  // Copy ctable and dctable
  for (int i = 0; i < ntable; i++) {
    h_table(i) = ctable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_ctable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  for (int i = 0; i < ntable; i++) {
    h_table(i) = dctable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_dctable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  // Copy etable and detable
  for (int i = 0; i < ntable; i++) {
    h_table(i) = etable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_etable = d_table;
  }

  {
  host_table_type h_table("HostTable",ntable);
  table_type d_table("DeviceTable",ntable);

  for (int i = 0; i < ntable; i++) {
    h_table(i) = detable[i];
  }
  Kokkos::deep_copy(d_table,h_table);
  d_detable = d_table;
  }
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCharmmCoulLongKokkos<DeviceType>::init_style()
{
  PairLJCharmmCoulLong::init_style();

  Kokkos::deep_copy(d_cut_ljsq,cut_ljsq);
  Kokkos::deep_copy(d_cut_coulsq,cut_coulsq);

  // error if rRESPA with inner levels

  if (update->whichflag == 1 && utils::strmatch(update->integrate_style,"^respa")) {
    int respa = 0;
    if (((Respa *) update->integrate)->level_inner >= 0) respa = 1;
    if (((Respa *) update->integrate)->level_middle >= 0) respa = 2;
    if (respa)
      error->all(FLERR,"Cannot use Kokkos pair style with rRESPA inner/middle");
  }

  // adjust neighbor list request for KOKKOS

  neighflag = lmp->kokkos->neighflag;
  auto request = neighbor->find_request(this);
  request->set_kokkos_host(std::is_same_v<DeviceType,LMPHostType> &&
                           !std::is_same_v<DeviceType,LMPDeviceType>);
  request->set_kokkos_device(std::is_same_v<DeviceType,LMPDeviceType>);
  if (neighflag == FULL) request->enable_full();
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

template<class DeviceType>
double PairLJCharmmCoulLongKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairLJCharmmCoulLong::init_one(i,j);

  k_params.h_view(i,j).lj1 = lj1[i][j];
  k_params.h_view(i,j).lj2 = lj2[i][j];
  k_params.h_view(i,j).lj3 = lj3[i][j];
  k_params.h_view(i,j).lj4 = lj4[i][j];
  //k_params.h_view(i,j).offset = offset[i][j];
  k_params.h_view(i,j).cut_ljsq = cut_ljsq;
  k_params.h_view(i,j).cut_coulsq = cut_coulsq;

  k_params.h_view(j,i) = k_params.h_view(i,j);
  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
    m_cut_ljsq[j][i] = m_cut_ljsq[i][j] = cut_ljsq;
    m_cut_coulsq[j][i] = m_cut_coulsq[i][j] = cut_coulsq;
  }

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}

namespace LAMMPS_NS {
template class PairLJCharmmCoulLongKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLJCharmmCoulLongKokkos<LMPHostType>;
#endif
}
