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

#include "pair_coul_cut_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"

#include <cmath>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairCoulCutKokkos<DeviceType>::PairCoulCutKokkos(LAMMPS *lmp) : PairCoulCut(lmp)
{
  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | Q_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairCoulCutKokkos<DeviceType>::~PairCoulCutKokkos()
{
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq, cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairCoulCutKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
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
  k_cut_ljsq.template sync<DeviceType>();
  k_cut_coulsq.template sync<DeviceType>();
  k_params.template sync<DeviceType>();
  if (eflag || vflag) atomKK->modified(execution_space,datamask_modify);
  else atomKK->modified(execution_space,F_MASK);

  x = atomKK->k_x.view<DeviceType>();
  c_x = atomKK->k_x.view<DeviceType>();
  f = atomKK->k_f.view<DeviceType>();
  q = atomKK->k_q.view<DeviceType>();
  type = atomKK->k_type.view<DeviceType>();

  x_rel =atomKK->k_x_rel.view<DeviceType>();
  c_x_rel = atomKK->k_x_rel.view<DeviceType>();
  bin_base = atomKK->k_bin_base.view<DeviceType>();
  binsizex = atomKK->binsizex;
  binsizey = atomKK->binsizey;
  binsizez = atomKK->binsizez;
  binsizex_h = __double2half(binsizex);
  binsizey_h = __double2half(binsizey);
  binsizez_h = __double2half(binsizez);
  fhcut_split = atomKK->k_fhcut_split.view<DeviceType>();

  nlocal = atom->nlocal;
  nall = atom->nlocal + atom->nghost;
  newton_pair = force->newton_pair;
  special_lj[0] = force->special_lj[0];
  special_lj[1] = force->special_lj[1];
  special_lj[2] = force->special_lj[2];
  special_lj[3] = force->special_lj[3];
  special_coul[0] = force->special_coul[0];
  special_coul[1] = force->special_coul[1];
  special_coul[2] = force->special_coul[2];
  special_coul[3] = force->special_coul[3];
  qqrd2e = force->qqrd2e;

  // loop over neighbors of my atoms
  // printf("in PairCoulCutKokkos::compute\n");

  EV_FLOAT ev;
  if (this->prec_type == DEFAULT_PREC) {
    ev = pair_compute<PairCoulCutKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == DOUBLE_PREC) {
    ev = CoulKernels::pair_compute_custom<PairCoulCutKokkos<DeviceType>, DOUBLE_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == FLOAT_PREC) {
    ev = CoulKernels::pair_compute_custom<PairCoulCutKokkos<DeviceType>, FLOAT_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == HALF_PREC) {
    ev = CoulKernels::pair_compute_custom<PairCoulCutKokkos<DeviceType>, HALF_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == HFMIX_PREC) {
    ev = CoulKernels::pair_compute_custom<PairCoulCutKokkos<DeviceType>, HFMIX_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else {
    // other precision not supported
    error->all(FLERR,"Invalid precision type");
  }

  Kokkos::fence();

  if (eflag) eng_coul += ev.ecoul;
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
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairCoulCutKokkos<DeviceType>::
compute_fcoul(const F_FLOAT& rsq, const int& /*i*/, const int&j, const int& itype,
              const int& jtype, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT rinv = sqrt(r2inv);
  F_FLOAT forcecoul;

  forcecoul = qqrd2e*(STACKPARAMS?m_params[itype][jtype].scale:params(itype,jtype).scale)*
    qtmp *q(j) *rinv;

  return factor_coul*forcecoul*r2inv;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairCoulCutKokkos<DeviceType>::
compute_ecoul(const F_FLOAT& rsq, const int& /*i*/, const int&j, const int& itype,
              const int& jtype, const F_FLOAT& factor_coul, const F_FLOAT& qtmp) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT rinv = sqrt(r2inv);

  return factor_coul*qqrd2e * (STACKPARAMS?m_params[itype][jtype].scale:params(itype,jtype).scale)
    * qtmp *q(j)*rinv;
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCoulCutKokkos<DeviceType>::allocate()
{
  PairCoulCut::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();

  k_cut_ljsq = typename ArrayTypes<DeviceType>::tdual_ffloat_2d("pair:cut_ljsq",n+1,n+1);
  d_cut_ljsq = k_cut_ljsq.template view<DeviceType>();
  k_cut_coulsq = typename ArrayTypes<DeviceType>::tdual_ffloat_2d("pair:cut_coulsq",n+1,n+1);
  d_cut_coulsq = k_cut_coulsq.template view<DeviceType>();

  k_params = Kokkos::DualView<params_coul**,Kokkos::LayoutRight,DeviceType>("PairCoulCut::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCoulCutKokkos<DeviceType>::settings(int narg, char **arg)
{
  // \todo check what should be the limit on narg
  if (narg > 2) error->all(FLERR,"Illegal pair_style command");

  PairCoulCut::settings(1,arg);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairCoulCutKokkos<DeviceType>::init_style()
{
  PairCoulCut::init_style();

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
double PairCoulCutKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairCoulCut::init_one(i,j);

  k_params.h_view(i,j).scale = scale[i][j];
  k_params.h_view(i,j).cutsq = cutone*cutone;
  k_params.h_view(j,i) = k_params.h_view(i,j);

  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
    m_cut_ljsq[j][i] = m_cut_ljsq[i][j] = cutone*cutone;
    m_cut_coulsq[j][i] = m_cut_coulsq[i][j] = cutone*cutone;
  }
  k_cutsq.h_view(i,j) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_cut_ljsq.h_view(i,j) = cutone*cutone;
  k_cut_ljsq.template modify<LMPHostType>();
  k_cut_coulsq.h_view(i,j) = cutone*cutone;
  k_cut_coulsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  return cutone;
}



namespace LAMMPS_NS {
template class PairCoulCutKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairCoulCutKokkos<LMPHostType>;
#endif
}

