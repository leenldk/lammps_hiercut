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

#include "pair_lj_cut_kokkos.h"

#include "atom_kokkos.h"
#include "atom_masks.h"
#include "error.h"
#include "force.h"
#include "kokkos.h"
#include "memory_kokkos.h"
#include "neigh_request.h"
#include "neighbor.h"
#include "respa.h"
#include "update.h"

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutKokkos<DeviceType>::PairLJCutKokkos(LAMMPS *lmp) : PairLJCut(lmp)
{
  respa_enable = 0;

  kokkosable = 1;
  atomKK = (AtomKokkos *) atom;
  execution_space = ExecutionSpaceFromDevice<DeviceType>::space;
  datamask_read = X_MASK | F_MASK | TYPE_MASK | ENERGY_MASK | VIRIAL_MASK;
  datamask_modify = F_MASK | ENERGY_MASK | VIRIAL_MASK;
  
  printf("atom nlocal : %d, nghost : %d, nmax : %d\n", atom->nlocal, atom->nghost, atom->nmax);
  //ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", atom->nlocal);

  pair_compute_time = 0;
  cuda_kernel_time = 0;
  init_time = 0;

  printf("in PairLJCutKokkos::PairLJCutKokkos, addr %#lx\n", this);
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
PairLJCutKokkos<DeviceType>::~PairLJCutKokkos()
{
  //printf("destroy PairLJCutKokkos, addr %#lx\n", this);
  if (copymode) return;

  if (allocated) {
    memoryKK->destroy_kokkos(k_eatom,eatom);
    memoryKK->destroy_kokkos(k_vatom,vatom);
    memoryKK->destroy_kokkos(k_cutsq,cutsq);
  }
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutKokkos<DeviceType>::compute(int eflag_in, int vflag_in)
{
  // printf("in PairLJCutKokkos::compute\n");
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

  // loop over neighbors of my atoms

  copymode = 1;

  Kokkos::Timer pair_compute_timer;
  pair_compute_timer.reset();
  EV_FLOAT ev;
  if (this->prec_type == DEFAULT_PREC) {
    ev = pair_compute<PairLJCutKokkos<DeviceType>,void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == DOUBLE_PREC) {
    ev = LJKernelsExpr::pair_compute_custom<PairLJCutKokkos<DeviceType>, DOUBLE_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
    //ev = LJKernels::pair_compute_custom<PairLJCutKokkos<DeviceType>, DOUBLE_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == FLOAT_PREC) {
    ev = LJKernelsExpr::pair_compute_custom<PairLJCutKokkos<DeviceType>, FLOAT_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
    //ev = LJKernels::pair_compute_custom<PairLJCutKokkos<DeviceType>, FLOAT_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == HALF_PREC) {
    ev = LJKernelsExpr::pair_compute_custom<PairLJCutKokkos<DeviceType>, HALF_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
    //ev = LJKernels::pair_compute_custom<PairLJCutKokkos<DeviceType>, HALF_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else if (this->prec_type == HFMIX_PREC) {
    ev = LJKernelsExpr::pair_compute_custom<PairLJCutKokkos<DeviceType>, HFMIX_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
    //ev = LJKernels::pair_compute_custom<PairLJCutKokkos<DeviceType>, HFMIX_PREC, void >(this,(NeighListKokkos<DeviceType>*)list);
  }
  else {
    // other precision not supported
    error->all(FLERR,"Invalid precision type");
  }
  
  
  
  Kokkos::fence();
  pair_compute_time += pair_compute_timer.seconds();

  if (eflag_global) eng_vdwl += ev.evdwl;
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
void PairLJCutKokkos<DeviceType>::summary()
{
  printf("PairLJCutKokkos::summary\n");
  printf("PairLJCutKokkos::pair_compute_time = %lf\n", pair_compute_time);
  printf("PairLJCutKokkos::cuda_kernel_time = %lf\n", cuda_kernel_time);
  printf("PairLJCutKokkos::init_time = %lf\n", init_time);
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCutKokkos<DeviceType>::
compute_fpair(const F_FLOAT &rsq, const int &, const int &, const int &itype, const int &jtype) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT r6inv = r2inv*r2inv*r2inv;

  const F_FLOAT forcelj = r6inv *
    ((STACKPARAMS?m_params[itype][jtype].lj1:params(itype,jtype).lj1)*r6inv -
     (STACKPARAMS?m_params[itype][jtype].lj2:params(itype,jtype).lj2));

  return forcelj*r2inv;
}

template<class DeviceType>
template<bool STACKPARAMS, class Specialisation>
KOKKOS_INLINE_FUNCTION
F_FLOAT PairLJCutKokkos<DeviceType>::
compute_evdwl(const F_FLOAT &rsq, const int &, const int &, const int &itype, const int &jtype) const {
  const F_FLOAT r2inv = 1.0/rsq;
  const F_FLOAT r6inv = r2inv*r2inv*r2inv;

  return r6inv*((STACKPARAMS?m_params[itype][jtype].lj3:params(itype,jtype).lj3)*r6inv -
                (STACKPARAMS?m_params[itype][jtype].lj4:params(itype,jtype).lj4)) -
                (STACKPARAMS?m_params[itype][jtype].offset:params(itype,jtype).offset);
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutKokkos<DeviceType>::allocate()
{
  PairLJCut::allocate();

  int n = atom->ntypes;
  memory->destroy(cutsq);
  memoryKK->create_kokkos(k_cutsq,cutsq,n+1,n+1,"pair:cutsq");
  d_cutsq = k_cutsq.template view<DeviceType>();
  k_params = Kokkos::DualView<params_lj**,Kokkos::LayoutRight,DeviceType>("PairLJCut::params",n+1,n+1);
  params = k_params.template view<DeviceType>();
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutKokkos<DeviceType>::settings(int narg, char **arg)
{
  if (narg > 2) error->all(FLERR,"Illegal pair_style command");

  PairLJCut::settings(1,arg);
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

template<class DeviceType>
void PairLJCutKokkos<DeviceType>::init_style()
{
  PairLJCut::init_style();

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
double PairLJCutKokkos<DeviceType>::init_one(int i, int j)
{
  double cutone = PairLJCut::init_one(i,j);

  k_params.h_view(i,j).lj1 = lj1[i][j];
  k_params.h_view(i,j).lj2 = lj2[i][j];
  k_params.h_view(i,j).lj3 = lj3[i][j];
  k_params.h_view(i,j).lj4 = lj4[i][j];
  k_params.h_view(i,j).offset = offset[i][j];
  k_params.h_view(i,j).cutsq = cutone*cutone;
  k_params.h_view(j,i) = k_params.h_view(i,j);
  if (i<MAX_TYPES_STACKPARAMS+1 && j<MAX_TYPES_STACKPARAMS+1) {
    m_params[i][j] = m_params[j][i] = k_params.h_view(i,j);
    m_cutsq[j][i] = m_cutsq[i][j] = cutone*cutone;
  }

  k_cutsq.h_view(i,j) = k_cutsq.h_view(j,i) = cutone*cutone;
  k_cutsq.template modify<LMPHostType>();
  k_params.template modify<LMPHostType>();

  if(i == 1 && j == 1) {
    printf("PairLJCutKokkos init m_cutsq[1][1] = %lf\n", m_cutsq[1][1]);
  }

  return cutone;
}



namespace LAMMPS_NS {
template class PairLJCutKokkos<LMPDeviceType>;
#ifdef LMP_KOKKOS_GPU
template class PairLJCutKokkos<LMPHostType>;
#endif
}

