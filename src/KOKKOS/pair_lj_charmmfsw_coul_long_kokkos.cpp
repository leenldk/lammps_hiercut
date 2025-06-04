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
   Contributing author: Mitch Murphy (alphataubio@gmail.com)

   Based on serial kspace lj-fsw sections (force-switched) provided by
   Robert Meissner and Lucio Colombi Ciacchi of Bremen University, Germany,
   with additional assistance from Robert A. Latour, Clemson University

 ------------------------------------------------------------------------- */

#include "pair_lj_charmmfsw_coul_long_kokkos.h"

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
PairLJCharmmfswCoulLongKokkos<DeviceType>::PairLJCharmmfswCoulLongKokkos(LAMMPS *lmp):PairLJCharmmfswCoulLong(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | Q_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCharmmfswCoulLongKokkos<DeviceType>::~PairLJCharmmfswCoulLongKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCharmmfswCoulLongKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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

  if (!param_inited) {
    param_inited = true;
    {
      int ntypes = atom->ntypes;
      cudaMalloc((void**)&param_lj12_f, ntypes * ntypes * sizeof(float2));
      cudaMalloc((void**)&param_lj34_f, ntypes * ntypes * sizeof(float2));
      cudaMalloc((void**)&param_lj12, ntypes * ntypes * sizeof(double2));
      cudaMalloc((void**)&param_lj34, ntypes * ntypes * sizeof(double2));
      
      auto t_params = params;
      auto t_param_lj12_f = param_lj12_f;
      auto t_param_lj34_f = param_lj34_f;
      auto t_param_lj12 = param_lj12;
      auto t_param_lj34 = param_lj34;

      using MDRangePolicy = Kokkos::MDRangePolicy<DeviceType,
          Kokkos::IndexType<int>,
          Kokkos::Rank<2, Kokkos::Iterate::Left, Kokkos::Iterate::Left> >;
      typename MDRangePolicy::point_type lower_bound = {1, 1};
      typename MDRangePolicy::point_type upper_bound = {ntypes + 1, ntypes + 1};
      MDRangePolicy policy(lower_bound, upper_bound);

      Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const int i, const int j) {
        t_param_lj12_f[(i - 1) * ntypes + (j - 1)].x = (float)(t_params(i, j).lj1);
        t_param_lj12_f[(i - 1) * ntypes + (j - 1)].y = (float)(t_params(i, j).lj2);
        t_param_lj34_f[(i - 1) * ntypes + (j - 1)].x = (float)(t_params(i, j).lj3);
        t_param_lj34_f[(i - 1) * ntypes + (j - 1)].y = (float)(t_params(i, j).lj4);
        t_param_lj12[(i - 1) * ntypes + (j - 1)].x = t_params(i, j).lj1;
        t_param_lj12[(i - 1) * ntypes + (j - 1)].y = t_params(i, j).lj2;
        t_param_lj34[(i - 1) * ntypes + (j - 1)].x = t_params(i, j).lj3;
        t_param_lj34[(i - 1) * ntypes + (j - 1)].y = t_params(i, j).lj4;
      });
      Kokkos::fence();
    }
    {
      int table_size = d_rtable.extent(0);
      int coul_table_size = 1<<16;
      printf("init float d_table, table size : %d\n", table_size);

      cudaMalloc((void**)&coul_ftable_f, coul_table_size * sizeof(float));

      cudaMalloc((void**)&d_rtable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_drtable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_ftable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_dftable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_ctable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_dctable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_etable_f, table_size * sizeof(float));
      cudaMalloc((void**)&d_detable_f, table_size * sizeof(float));

      auto t_d_rtable = d_rtable;
      auto t_d_drtable = d_drtable;
      auto t_d_ftable = d_ftable;
      auto t_d_dftable = d_dftable;
      auto t_d_ctable = d_ctable;
      auto t_d_dctable = d_dctable;
      auto t_d_etable = d_etable;
      auto t_d_detable = d_detable;

      auto t_d_rtable_f = d_rtable_f;
      auto t_d_drtable_f = d_drtable_f;
      auto t_d_ftable_f = d_ftable_f;
      auto t_d_dftable_f = d_dftable_f;
      auto t_d_ctable_f = d_ctable_f;
      auto t_d_dctable_f = d_dctable_f;
      auto t_d_etable_f = d_etable_f;
      auto t_d_detable_f = d_detable_f;

      auto t_coul_ftable_f = coul_ftable_f;
      auto t_ncoulmask = ncoulmask;
      auto t_ncoulshiftbits = ncoulshiftbits;

      Kokkos::parallel_for(table_size, KOKKOS_LAMBDA (const int i) {
        t_d_rtable_f[i] = (float)t_d_rtable[i];
        t_d_drtable_f[i] = (float)t_d_drtable[i];
        t_d_ftable_f[i] = (float)t_d_ftable[i];
        t_d_dftable_f[i] = (float)t_d_dftable[i];
        t_d_ctable_f[i] = (float)t_d_ctable[i];
        t_d_dctable_f[i] = (float)t_d_dctable[i];
        t_d_etable_f[i] = (float)t_d_etable[i];
        t_d_detable_f[i] = (float)t_d_detable[i];
      });
      Kokkos::fence();

      printf("before init coul_ftable\n"); fflush(stdout);
      Kokkos::parallel_for(coul_table_size, KOKKOS_LAMBDA (const int i) {
        __half rsq_h = __ushort_as_half(static_cast<ushort>(i));
        float rsq_f = __half2float(rsq_h);
        double rsq_d = (double)rsq_f;

        union_int_float_t rsq_lookup;
        rsq_lookup.f = rsq_f;
        const int itable = (rsq_lookup.i & t_ncoulmask) >> t_ncoulshiftbits;

        const double fraction = (rsq_d - t_d_rtable[itable]) * t_d_drtable[itable];
        const double table = t_d_ftable[itable] + fraction * t_d_dftable[itable];
        const float f1_rsq_f = table / rsq_d;

        t_coul_ftable_f[i] = f1_rsq_f;
      });
      Kokkos::fence();

      printf("after init coul_ftable\n"); fflush(stdout);
    }
  }

  // static int epoch = 0;
  // if (epoch % 210 == 0)
  // {
  //   std::string file_name = "neigh_size" + std::to_string(epoch) + ".txt";
  //   FILE* file = fopen(file_name.c_str(), "w");
  //   for(int i = 0; i < list->inum; i++) {
  //     fprintf(file, "%d\n", ((NeighListKokkos<DeviceType>*)list) -> d_numneigh[i]);
  //   }
  //   fclose(file);
  // }
  // epoch++;

  // loop over neighbors of my atoms

  copymode = 1;

  EV_FLOAT ev;
  // ncoultablebits != 0
  if (this->prec_type == DEFAULT_PREC) {
    // use default precision
    if (ncoultablebits)
      ev = pair_compute<PairLJCharmmfswCoulLongKokkos<DeviceType>,CoulLongTable<1> >
        (this,(NeighListKokkos<DeviceType>*)list);
    else
      ev = pair_compute<PairLJCharmmfswCoulLongKokkos<DeviceType>,CoulLongTable<0> >
        (this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == DOUBLE_PREC) {
    ev = CharmmfswCoulLongKernels::pair_compute_custom<PairLJCharmmfswCoulLongKokkos<DeviceType>, DOUBLE_PREC, CoulLongTable<1> >
        (this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == FLOAT_PREC) {
    ev = CharmmfswCoulLongKernels::pair_compute_custom<PairLJCharmmfswCoulLongKokkos<DeviceType>, FLOAT_PREC, CoulLongTable<1> >
        (this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == HFMIX_PREC) {
    ev = CharmmfswCoulLongKernels::pair_compute_custom<PairLJCharmmfswCoulLongKokkos<DeviceType>, HFMIX_PREC, CoulLongTable<1> >
        (this,(NeighListKokkos<DeviceType>*)list);
  }
  else {
    // other precision not supported
    error->all(FLERR,"Invalid precision type");
  }  


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

/* ----------------------------------------------------------------------
   compute LJ CHARMM pair force between atoms i and j
   ---------------------------------------------------------------------- */
template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCharmmfswCoulLongKokkos<DeviceType>::
compute_fpair(const F_FLOAT& rsq, const int& /*i*/, const int& /*j*/,
              const int& itype, const int& jtype) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT r6inv = r2inv*r2inv*r2inv;
  F_FLOAT forcelj, switch1;

  forcelj = r6inv *
    ((STACKPARAMS?m_params[itype][jtype].lj1:params(itype,jtype).lj1)*r6inv -
     (STACKPARAMS?m_params[itype][jtype].lj2:params(itype,jtype).lj2));

  if (rsq > cut_lj_innersq) {
    switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
              (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
    forcelj = forcelj*switch1;
  }

  return forcelj*r2inv;
}

/* ----------------------------------------------------------------------
   compute LJ CHARMM pair potential energy between atoms i and j
   ---------------------------------------------------------------------- */
template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCharmmfswCoulLongKokkos<DeviceType>::
compute_evdwl(const F_FLOAT& rsq, const int& /*i*/, const int& /*j*/,
              const int& itype, const int& jtype) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT r6inv = r2inv*r2inv*r2inv;
  const F_FLOAT r = sqrt(rsq);
  const F_FLOAT rinv = 1.0/r;
  const F_FLOAT r3inv = rinv*rinv*rinv;
  F_FLOAT englj, englj12, englj6;

  if (rsq > cut_lj_innersq) {
    englj12 = (STACKPARAMS?m_params[itype][jtype].lj3:params(itype,jtype).lj3)*cut_lj6*
      denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
    englj6 = -(STACKPARAMS?m_params[itype][jtype].lj4:params(itype,jtype).lj4)*
      cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
    englj = englj12 + englj6;
  } else {
    englj12 = r6inv*(STACKPARAMS?m_params[itype][jtype].lj3:params(itype,jtype).lj3)*r6inv -
    (STACKPARAMS?m_params[itype][jtype].lj3:params(itype,jtype).lj3)*cut_lj_inner6inv*cut_lj6inv;
    englj6 = -(STACKPARAMS?m_params[itype][jtype].lj4:params(itype,jtype).lj4)*r6inv +
      (STACKPARAMS?m_params[itype][jtype].lj4:params(itype,jtype).lj4)*
      cut_lj_inner3inv*cut_lj3inv;
    englj = englj12 + englj6;
  }
  return englj;
}

/* ----------------------------------------------------------------------
   compute coulomb pair force between atoms i and j
   ---------------------------------------------------------------------- */
template<class DeviceType>
template<bool STACKPARAMS,  class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCharmmfswCoulLongKokkos<DeviceType>::
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
F_FLOAT PairLJCharmmfswCoulLongKokkos<DeviceType>::
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
void PairLJCharmmfswCoulLongKokkos<DeviceType>::allocate()
{
  PairLJCharmmfswCoulLong::allocate();

  int n = atom->ntypes;

  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  d_cut_ljsq = typename AT::t_ffloat_2d("pair:cut_ljsq",n+1,n+1);

  d_cut_coulsq = typename AT::t_ffloat_2d("pair:cut_coulsq",n+1,n+1);

  k_params = Kokkos::DualView<params_lj_coul**,Kokkos::LayoutRight,DeviceType>("PairLJCharmmfswCoulLong::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

template<class DeviceType>
void PairLJCharmmfswCoulLongKokkos<DeviceType>::init_tables(double cut_coul, double *cut_respa)
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
void PairLJCharmmfswCoulLongKokkos<DeviceType>::init_style()
{
  PairLJCharmmfswCoulLong::init_style();

  Kokkos::deep_copy(d_cut_ljsq,cut_ljsq);
  Kokkos::deep_copy(d_cut_coulsq,cut_coulsq);
  cut_sq = MAX(cut_ljsq,cut_coulsq);

  printf("cut_ljsq : %f, cut_coulsq : %f, cut_sq : %f\n", cut_ljsq, cut_coulsq, cut_sq);

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
double PairLJCharmmfswCoulLongKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairLJCharmmfswCoulLong::init_one(i,j);

  k_params.h_view(i,j).lj1 = lj1[i][j];
  k_params.h_view(i,j).lj2 = lj2[i][j];
  k_params.h_view(i,j).lj3 = lj3[i][j];
  k_params.h_view(i,j).lj4 = lj4[i][j];
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
template class PairLJCharmmfswCoulLongKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLJCharmmfswCoulLongKokkos<LMPHostType>;
#endif
}
