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

#else

// clang-format off
#ifndef LMP_PAIR_KOKKOS_H
#define LMP_PAIR_KOKKOS_H

#include "pair.h"               // IWYU pragma: export
#include "neighbor_kokkos.h"
#include "neigh_list_kokkos.h"
#include "math_special.h"
#include "update.h"
#include "Kokkos_Macros.hpp"
#include "Kokkos_ScatterView.hpp"

namespace LAMMPS_NS {

template<int Table>
struct CoulLongTable {
  enum {DoTable = Table};
};

// Tags for doing coulomb calculations or not
// They facilitate function overloading, since
// partial template specialization of member functions is not allowed
struct CoulTag {};
struct NoCoulTag {};

template<int FLAG>
struct DoCoul {
  typedef NoCoulTag type;
};

template<>
struct DoCoul<1> {
  typedef CoulTag type;
};


//Specialisation for Neighborlist types Half, HalfThread, Full
template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctor  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  // The copy of the pair style
  PairStyle c;
  typename AT::t_f_array f;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;
  int inum;

  using KKDeviceType = typename KKDevice<device_type>::value;
  using DUP = NeedDup_v<NEIGHFLAG,device_type>;

  // The force array is atomic for Half/Thread neighbor style
  //Kokkos::View<F_FLOAT*[3], typename DAT::t_f_array::array_layout,
  //             typename KKDevice<device_type>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > f;
  KKScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,KKDeviceType,KKScatterSum,DUP> dup_f;

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  //Kokkos::View<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,
  //             typename KKDevice<device_type>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > eatom;
  KKScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,KKDeviceType,KKScatterSum,DUP> dup_eatom;

  //Kokkos::View<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,
  //             typename KKDevice<device_type>::value,Kokkos::MemoryTraits<AtomicF<NEIGHFLAG>::value> > vatom;
  KKScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,KKDeviceType,KKScatterSum,DUP> dup_vatom;

  NeighListKokkos<device_type> list;

  PairComputeFunctor(PairStyle* c_ptr,
                          NeighListKokkos<device_type>* list_ptr):
  c(*c_ptr),list(*list_ptr) {
    // allocate duplicated memory
    f = c.f;
    d_eatom = c.d_eatom;
    d_vatom = c.d_vatom;
    dup_f     = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.d_vatom);
    inum = list.inum;
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctor() {c.copymode = 1; list.copymode = 1;};

  KOKKOS_INLINE_FUNCTION int sbmask(const int& j) const {
    return j >> SBBITS & 3;
  }

  void contribute() {
    int need_dup = std::is_same_v<DUP,Kokkos::Experimental::ScatterDuplicated>;

    if (need_dup) {
      Kokkos::Experimental::contribute(c.f, dup_f);

      if (c.eflag_atom)
        Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

      if (c.vflag_atom)
        Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
    }
  }

  // Loop over neighbors of one atom without coulomb interaction
  // This function is called in parallel

  template<int EVFLAG, int NEWTON_PAIR>
  KOKKOS_FUNCTION
  EV_FLOAT compute_item(const int& ii,
                        const NeighListKokkos<device_type> &list, const NoCoulTag&) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    EV_FLOAT ev;
    const int i = list.d_ilist[ii];
    const X_FLOAT xtmp = c.x(i,0);
    const X_FLOAT ytmp = c.x(i,1);
    const X_FLOAT ztmp = c.x(i,2);
    const int itype = c.type(i);

    const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
    const int jnum = list.d_numneigh[i];

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    if (NEIGHFLAG == FULL && ZEROFLAG) {
      f(i,0) = 0.0;
      f(i,1) = 0.0;
      f(i,2) = 0.0;
    }

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

        const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        }

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (c.eflag_either) {
            evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            ev.evdwl += (((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD)&&(NEWTON_PAIR||(j<c.nlocal)))?1.0:0.5)*evdwl;
          }

          if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl,fpair,delx,dely,delz);
        }
      }

    }

    a_f(i,0) += fxtmp;
    a_f(i,1) += fytmp;
    a_f(i,2) += fztmp;

    return ev;
  }

  // Loop over neighbors of one atom with coulomb interaction
  // This function is called in parallel

  template<int EVFLAG, int NEWTON_PAIR>
  KOKKOS_FUNCTION
  EV_FLOAT compute_item(const int& ii,
                        const NeighListKokkos<device_type> &list, const CoulTag& ) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    EV_FLOAT ev;
    const int i = list.d_ilist[ii];
    const X_FLOAT xtmp = c.x(i,0);
    const X_FLOAT ytmp = c.x(i,1);
    const X_FLOAT ztmp = c.x(i,2);
    const int itype = c.type(i);
    const F_FLOAT qtmp = c.q(i);

    const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
    const int jnum = list.d_numneigh[i];

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    if (NEIGHFLAG == FULL && ZEROFLAG) {
      f(i,0) = 0.0;
      f(i,1) = 0.0;
      f(i,2) = 0.0;
    }

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype)))
          fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
        if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype)))
          fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        }

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag_either) {
            if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype))) {
              evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              ev.evdwl += (((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && (NEWTON_PAIR || (j < c.nlocal)))?1.0:0.5)*evdwl;
            }
            if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype))) {
              ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              ev.ecoul += (((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && (NEWTON_PAIR || (j < c.nlocal)))?1.0:0.5)*ecoul;
            }
          }

          if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl+ecoul,fpair,delx,dely,delz);
        }
      }
    }

    a_f(i,0) += fxtmp;
    a_f(i,1) += fytmp;
    a_f(i,2) += fztmp;

    return ev;
  }

  // TeamPolicy, newton off, and no energy/virial
  // Loop over neighbors of one atom without coulomb interaction
  // This function is called in parallel

  KOKKOS_FUNCTION
  void compute_item_team(typename Kokkos::TeamPolicy<device_type>::member_type team,
                         const NeighListKokkos<device_type> &list, const NoCoulTag&) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    const int atoms_per_team = team.team_size();
    const int firstatom = team.league_rank()*atoms_per_team;
    const int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);

      if (NEIGHFLAG == FULL && ZEROFLAG) {
        Kokkos::single(Kokkos::PerThread(team), [&] (){
          f(i,0) = 0.0;
          f(i,1) = 0.0;
          f(i,2) = 0.0;
        });
      }

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      t_scalar3<double> fsum;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, t_scalar3<double>& ftmp) {

        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);

          const F_FLOAT fx = delx*fpair;
          const F_FLOAT fy = dely*fpair;
          const F_FLOAT fz = delz*fpair;

          ftmp.x += fx;
          ftmp.y += fy;
          ftmp.z += fz;

          if ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && j < c.nlocal) {
            a_f(j,0) -= fx;
            a_f(j,1) -= fy;
            a_f(j,2) -= fz;
          }
        }

      },fsum);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
        a_f(i,0) += fsum.x;
        a_f(i,1) += fsum.y;
        a_f(i,2) += fsum.z;
      });

    });
  }

  // TeamPolicy, newton off, and no energy/virial
  // Loop over neighbors of one atom with coulomb interaction
  // This function is called in parallel

  KOKKOS_FUNCTION
  void compute_item_team(typename Kokkos::TeamPolicy<device_type>::member_type team,
                         const NeighListKokkos<device_type> &list, const CoulTag& ) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    const int atoms_per_team = team.team_size();
    int firstatom = team.league_rank()*atoms_per_team;
    int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);
      const F_FLOAT qtmp = c.q(i);

      if (NEIGHFLAG == FULL && ZEROFLAG) {
        Kokkos::single(Kokkos::PerThread(team), [&] ()
        {
          f(i,0) = 0.0;
          f(i,1) = 0.0;
          f(i,2) = 0.0;
        });
      }

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      t_scalar3<double> fsum;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, t_scalar3<double>& ftmp) {
        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          F_FLOAT fpair = F_FLOAT();

          if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype)))
            fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype)))
            fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

          const F_FLOAT fx = delx*fpair;
          const F_FLOAT fy = dely*fpair;
          const F_FLOAT fz = delz*fpair;

          ftmp.x += fx;
          ftmp.y += fy;
          ftmp.z += fz;

          if ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && j < c.nlocal) {
            a_f(j,0) -= fx;
            a_f(j,1) -= fy;
            a_f(j,2) -= fz;
          }
        }

      },fsum);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
        a_f(i,0) += fsum.x;
        a_f(i,1) += fsum.y;
        a_f(i,2) += fsum.z;
      });
    });
  }

  // TeamPolicy, newton off, and energy/virial
  // Loop over neighbors of one atom without coulomb interaction
  // This function is called in parallel

  KOKKOS_FUNCTION
  EV_FLOAT compute_item_team_ev(typename Kokkos::TeamPolicy<device_type>::member_type team,
                                const NeighListKokkos<device_type> &list, const NoCoulTag&) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    auto a_eatom = dup_eatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    auto a_vatom = dup_vatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    EV_FLOAT ev;

    const int atoms_per_team = team.team_size();
    const int firstatom = team.league_rank()*atoms_per_team;
    const int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);

      if (NEIGHFLAG == FULL && ZEROFLAG) {
        Kokkos::single(Kokkos::PerThread(team), [&] ()
        {
          f(i,0) = 0.0;
          f(i,1) = 0.0;
          f(i,2) = 0.0;
        });
      }

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      FEV_FLOAT fev;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, FEV_FLOAT& fev_tmp) {

        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);

          const F_FLOAT fx = delx*fpair;
          const F_FLOAT fy = dely*fpair;
          const F_FLOAT fz = delz*fpair;

          fev_tmp.f[0] += fx;
          fev_tmp.f[1] += fy;
          fev_tmp.f[2] += fz;

          const int I_CONTRIB = (NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD);
          const int J_CONTRIB = ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && j < c.nlocal);
          const E_FLOAT factor = J_CONTRIB?1.0:0.5;

          if (J_CONTRIB) {
            a_f(j,0) -= fx;
            a_f(j,1) -= fy;
            a_f(j,2) -= fz;
          }

          F_FLOAT evdwl = 0.0;
          if (c.eflag_either) {
            evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            fev_tmp.evdwl += factor * evdwl;

            if (c.eflag_atom) {
              const E_FLOAT epairhalf = 0.5 * evdwl;

              if (I_CONTRIB)
                a_eatom[i] += epairhalf;

              if (J_CONTRIB)
                a_eatom[j] += epairhalf;
            }
          }

          if (c.vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            fev_tmp.v[0] += factor*v0;
            fev_tmp.v[1] += factor*v1;
            fev_tmp.v[2] += factor*v2;
            fev_tmp.v[3] += factor*v3;
            fev_tmp.v[4] += factor*v4;
            fev_tmp.v[5] += factor*v5;

            if (c.vflag_atom) {
              if (I_CONTRIB) {
                a_vatom(i,0) += 0.5*v0;
                a_vatom(i,1) += 0.5*v1;
                a_vatom(i,2) += 0.5*v2;
                a_vatom(i,3) += 0.5*v3;
                a_vatom(i,4) += 0.5*v4;
                a_vatom(i,5) += 0.5*v5;
              }
              if (J_CONTRIB) {
                a_vatom(j,0) += 0.5*v0;
                a_vatom(j,1) += 0.5*v1;
                a_vatom(j,2) += 0.5*v2;
                a_vatom(j,3) += 0.5*v3;
                a_vatom(j,4) += 0.5*v4;
                a_vatom(j,5) += 0.5*v5;
              }
            }
          }
        }
      },fev);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
        a_f(i,0) += fev.f[0];
        a_f(i,1) += fev.f[1];
        a_f(i,2) += fev.f[2];

        if (c.eflag_global)
          ev.evdwl += fev.evdwl;

        if (c.vflag_global) {
          ev.v[0] += fev.v[0];
          ev.v[1] += fev.v[1];
          ev.v[2] += fev.v[2];
          ev.v[3] += fev.v[3];
          ev.v[4] += fev.v[4];
          ev.v[5] += fev.v[5];
        }

        if (NEIGHFLAG == FULL) {

          if (c.eflag_atom)
            a_eatom(i) += fev.evdwl;

          if (c.vflag_atom) {
            a_vatom(i,0) += fev.v[0];
            a_vatom(i,1) += fev.v[1];
            a_vatom(i,2) += fev.v[2];
            a_vatom(i,3) += fev.v[3];
            a_vatom(i,4) += fev.v[4];
            a_vatom(i,5) += fev.v[5];
          }
        }
      });
    });
    return ev;
  }

  // TeamPolicy, newton off, and energy/virial
  // Loop over neighbors of one atom with coulomb interaction
  // This function is called in parallel

  KOKKOS_FUNCTION
  EV_FLOAT compute_item_team_ev(typename Kokkos::TeamPolicy<device_type>::member_type team,
                                const NeighListKokkos<device_type> &list, const CoulTag& ) const {

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    auto a_eatom = dup_eatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    auto a_vatom = dup_vatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    EV_FLOAT ev;

    const int atoms_per_team = team.team_size();
    const int firstatom = team.league_rank()*atoms_per_team;
    const int lastatom = firstatom + atoms_per_team < inum ? firstatom + atoms_per_team : inum;
    Kokkos::parallel_for(Kokkos::TeamThreadRange(team, firstatom, lastatom), [&] (const int &ii) {

      const int i = list.d_ilist[ii];
      const X_FLOAT xtmp = c.x(i,0);
      const X_FLOAT ytmp = c.x(i,1);
      const X_FLOAT ztmp = c.x(i,2);
      const int itype = c.type(i);
      const F_FLOAT qtmp = c.q(i);

      if (NEIGHFLAG == FULL && ZEROFLAG) {
        Kokkos::single(Kokkos::PerThread(team), [&] (){
          f(i,0) = 0.0;
          f(i,1) = 0.0;
          f(i,2) = 0.0;
        });
      }

      const AtomNeighborsConst neighbors_i = list.get_neighbors_const(i);
      const int jnum = list.d_numneigh[i];

      FEV_FLOAT fev;

      Kokkos::parallel_reduce(Kokkos::ThreadVectorRange(team,jnum),
        [&] (const int jj, FEV_FLOAT& fev_tmp) {

        int j = neighbors_i(jj);
        const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
        const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
        j &= NEIGHMASK;
        const X_FLOAT delx = xtmp - c.x(j,0);
        const X_FLOAT dely = ytmp - c.x(j,1);
        const X_FLOAT delz = ztmp - c.x(j,2);
        const int jtype = c.type(j);
        const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq < (STACKPARAMS?c.m_cutsq[itype][jtype]:c.d_cutsq(itype,jtype))) {

          F_FLOAT fpair = F_FLOAT();

          if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype)))
            fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype)))
            fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

          const F_FLOAT fx = delx*fpair;
          const F_FLOAT fy = dely*fpair;
          const F_FLOAT fz = delz*fpair;

          fev_tmp.f[0] += fx;
          fev_tmp.f[1] += fy;
          fev_tmp.f[2] += fz;

          const int I_CONTRIB = (NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD);
          const int J_CONTRIB = ((NEIGHFLAG == HALF || NEIGHFLAG == HALFTHREAD) && j < c.nlocal);
          const E_FLOAT factor = J_CONTRIB?1.0:0.5;

          if (J_CONTRIB) {
            a_f(j,0) -= fx;
            a_f(j,1) -= fy;
            a_f(j,2) -= fz;
          }

          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag_either) {
            if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype))) {
              evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              fev_tmp.evdwl += factor * evdwl;
            }
            if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype))) {
              ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              fev_tmp.ecoul += factor * ecoul;
            }


            if (c.eflag_atom) {
              const E_FLOAT epairhalf = 0.5 * (evdwl + ecoul);

              if (I_CONTRIB)
                a_eatom[i] += epairhalf;

              if (J_CONTRIB)
                a_eatom[j] += epairhalf;
            }
          }

          if (c.vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            fev_tmp.v[0] += factor*v0;
            fev_tmp.v[1] += factor*v1;
            fev_tmp.v[2] += factor*v2;
            fev_tmp.v[3] += factor*v3;
            fev_tmp.v[4] += factor*v4;
            fev_tmp.v[5] += factor*v5;

            if (c.vflag_atom) {
              if (I_CONTRIB) {
                a_vatom(i,0) += 0.5*v0;
                a_vatom(i,1) += 0.5*v1;
                a_vatom(i,2) += 0.5*v2;
                a_vatom(i,3) += 0.5*v3;
                a_vatom(i,4) += 0.5*v4;
                a_vatom(i,5) += 0.5*v5;
              }
              if (J_CONTRIB) {
                a_vatom(j,0) += 0.5*v0;
                a_vatom(j,1) += 0.5*v1;
                a_vatom(j,2) += 0.5*v2;
                a_vatom(j,3) += 0.5*v3;
                a_vatom(j,4) += 0.5*v4;
                a_vatom(j,5) += 0.5*v5;
              }
            }
          }
        }
      },fev);

      Kokkos::single(Kokkos::PerThread(team), [&] () {
        a_f(i,0) += fev.f[0];
        a_f(i,1) += fev.f[1];
        a_f(i,2) += fev.f[2];

        if (c.eflag_global) {
          ev.evdwl += fev.evdwl;
          ev.ecoul += fev.ecoul;
        }

        if (c.vflag_global) {
          ev.v[0] += fev.v[0];
          ev.v[1] += fev.v[1];
          ev.v[2] += fev.v[2];
          ev.v[3] += fev.v[3];
          ev.v[4] += fev.v[4];
          ev.v[5] += fev.v[5];
        }

        if (NEIGHFLAG == FULL) {

          if (c.eflag_atom)
            a_eatom(i) += fev.evdwl + fev.ecoul;

          if (c.vflag_atom) {
            a_vatom(i,0) += fev.v[0];
            a_vatom(i,1) += fev.v[1];
            a_vatom(i,2) += fev.v[2];
            a_vatom(i,3) += fev.v[3];
            a_vatom(i,4) += fev.v[4];
            a_vatom(i,5) += fev.v[5];
          }
        }
      });
    });
    return ev;
  }

  KOKKOS_INLINE_FUNCTION
    void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    auto a_eatom = dup_eatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    auto a_vatom = dup_vatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    const int EFLAG = c.eflag_either;
    const int NEWTON_PAIR = c.newton_pair;
    const int VFLAG = c.vflag_either;

    if (EFLAG) {
      if (c.eflag_atom) {
        const E_FLOAT epairhalf = 0.5 * epair;
        if (NEWTON_PAIR || i < c.nlocal) a_eatom[i] += epairhalf;
        if ((NEWTON_PAIR || j < c.nlocal) && NEIGHFLAG != FULL) a_eatom[j] += epairhalf;
      }
    }

    if (VFLAG) {
      const E_FLOAT v0 = delx*delx*fpair;
      const E_FLOAT v1 = dely*dely*fpair;
      const E_FLOAT v2 = delz*delz*fpair;
      const E_FLOAT v3 = delx*dely*fpair;
      const E_FLOAT v4 = delx*delz*fpair;
      const E_FLOAT v5 = dely*delz*fpair;

      if (c.vflag_global) {
        if (NEIGHFLAG != FULL) {
          if (NEWTON_PAIR) {
            ev.v[0] += v0;
            ev.v[1] += v1;
            ev.v[2] += v2;
            ev.v[3] += v3;
            ev.v[4] += v4;
            ev.v[5] += v5;
          } else {
            if (i < c.nlocal) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }
            if (j < c.nlocal) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }
          }
        } else {
          ev.v[0] += 0.5*v0;
          ev.v[1] += 0.5*v1;
          ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3;
          ev.v[4] += 0.5*v4;
          ev.v[5] += 0.5*v5;
        }
      }

      if (c.vflag_atom) {
        if (NEWTON_PAIR || i < c.nlocal) {
          a_vatom(i,0) += 0.5*v0;
          a_vatom(i,1) += 0.5*v1;
          a_vatom(i,2) += 0.5*v2;
          a_vatom(i,3) += 0.5*v3;
          a_vatom(i,4) += 0.5*v4;
          a_vatom(i,5) += 0.5*v5;
        }
        if ((NEWTON_PAIR || j < c.nlocal) && NEIGHFLAG != FULL) {
          a_vatom(j,0) += 0.5*v0;
          a_vatom(j,1) += 0.5*v1;
          a_vatom(j,2) += 0.5*v2;
          a_vatom(j,3) += 0.5*v3;
          a_vatom(j,4) += 0.5*v4;
          a_vatom(j,5) += 0.5*v5;
        }
      }
    }
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    if (c.newton_pair) compute_item<0,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    else compute_item<0,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &energy_virial) const {
    if (c.newton_pair)
      energy_virial += compute_item<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    else
      energy_virial += compute_item<1,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::TeamPolicy<device_type>::member_type& team) const {
    compute_item_team(team,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const typename Kokkos::TeamPolicy<device_type>::member_type& team, value_type &energy_virial) const {
    energy_virial += compute_item_team_ev(team,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }
};


// Filter out Neighflags which are not supported for PairStyle
// The enable_if clause will invalidate the last parameter of the function, so that
// a match is only achieved, if PairStyle supports the specific neighborlist variant.
// This uses the fact that failure to match template parameters is not an error.
// By having the enable_if with a ! and without it, exactly one of the functions
// pair_compute_neighlist will match - either the dummy version
// or the real one further below

template<class PairStyle, unsigned NEIGHFLAG, int ZEROFLAG = 0, class Specialisation = void>
EV_FLOAT pair_compute_neighlist (PairStyle* fpair, std::enable_if_t<!((NEIGHFLAG&PairStyle::EnabledNeighFlags) != 0), NeighListKokkos<typename PairStyle::device_type>*> list) {
  EV_FLOAT ev;
  (void) fpair;
  (void) list;
  printf("ERROR: calling pair_compute with invalid neighbor list style: requested %i  available %i \n",NEIGHFLAG,PairStyle::EnabledNeighFlags);
  return ev;
}

template<class NeighStyle>
int GetMaxNeighs(NeighStyle* list)
{
  auto d_ilist = list->d_ilist;
  auto d_numneigh = list->d_numneigh;
  int inum = list->inum;

  int maxneigh = 0;
  Kokkos::parallel_reduce(inum, LAMMPS_LAMBDA(const int ii, int &maxneigh) {
    const int i = d_ilist[ii];
    const int num_neighs = d_numneigh[i];
    maxneigh = MAX(maxneigh,num_neighs);
  }, Kokkos::Max<int>(maxneigh));

  return maxneigh;
}

template<class DeviceType, class FunctorStyle>
void GetMaxTeamSize(FunctorStyle& functor, int inum,
                int &teamsize_max_for, int &teamsize_max_reduce)
{
  teamsize_max_for = Kokkos::TeamPolicy<DeviceType>(inum,Kokkos::AUTO).team_size_max(functor,Kokkos::ParallelForTag());
  teamsize_max_reduce = Kokkos::TeamPolicy<DeviceType>(inum,Kokkos::AUTO).team_size_max(functor,Kokkos::ParallelReduceTag());
}

// Submit ParallelFor for NEIGHFLAG=HALF,HALFTHREAD,FULL
template<class PairStyle, unsigned NEIGHFLAG, int ZEROFLAG = 0, class Specialisation = void>
EV_FLOAT pair_compute_neighlist (PairStyle* fpair, std::enable_if_t<(NEIGHFLAG&PairStyle::EnabledNeighFlags) != 0, NeighListKokkos<typename PairStyle::device_type>*> list) {
  EV_FLOAT ev;

  const int inum = list->inum;

  if (!fpair->lmp->kokkos->neigh_thread_set)
    if (fpair->lmp->kokkos->ngpus && inum <= 16000)
      if (NEIGHFLAG == FULL || !fpair->newton_pair)
        fpair->lmp->kokkos->neigh_thread = 1;

  if (fpair->lmp->kokkos->neigh_thread) {

    static int vectorsize = 0;
    static int atoms_per_team = 0;

#if defined(LMP_KOKKOS_GPU)
    static int teamsize_max_for = 0;
    static int teamsize_max_reduce = 0;
    static int lastcall = -1;
    if (!vectorsize || lastcall < fpair->lmp->neighbor->lastcall) {
      lastcall = fpair->lmp->update->ntimestep;
      vectorsize = GetMaxNeighs(list);
      vectorsize = MathSpecial::powint(2,(int(log2(vectorsize) + 0.5))); // round to nearest power of 2

  #if defined(KOKKOS_ENABLE_HIP)
      int max_vectorsize = 64;
  #else
      int max_vectorsize = 32;
  #endif

      vectorsize = MIN(vectorsize,max_vectorsize);

      if (fpair->atom->ntypes > MAX_TYPES_STACKPARAMS) {
        PairComputeFunctor<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list);
        GetMaxTeamSize<typename PairStyle::device_type>(ff, inum, teamsize_max_for, teamsize_max_reduce);
      } else {
        PairComputeFunctor<PairStyle,NEIGHFLAG,true,ZEROFLAG,Specialisation > ff(fpair,list);
        GetMaxTeamSize<typename PairStyle::device_type>(ff, inum, teamsize_max_for, teamsize_max_reduce);
      }
    }

    int teamsize_max = teamsize_max_for;
    if (fpair->eflag || fpair->vflag)
      teamsize_max = teamsize_max_reduce;
    atoms_per_team = teamsize_max/vectorsize;
#else
    vectorsize = 1;
    atoms_per_team = 1;
#endif

    const int num_teams = inum / atoms_per_team + (inum % atoms_per_team ? 1 : 0);

    if (fpair->atom->ntypes > MAX_TYPES_STACKPARAMS) {
      PairComputeFunctor<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list);
      Kokkos::TeamPolicy<typename PairStyle::device_type,Kokkos::IndexType<int> > policy(num_teams,atoms_per_team,vectorsize);
      if (fpair->eflag || fpair->vflag) Kokkos::parallel_reduce(policy,ff,ev);
      else                              Kokkos::parallel_for(policy,ff);
      ff.contribute();
    } else {
      PairComputeFunctor<PairStyle,NEIGHFLAG,true,ZEROFLAG,Specialisation > ff(fpair,list);
      Kokkos::TeamPolicy<typename PairStyle::device_type,Kokkos::IndexType<int> > policy(num_teams,atoms_per_team,vectorsize);
      if (fpair->eflag || fpair->vflag) Kokkos::parallel_reduce(policy,ff,ev);
      else                              Kokkos::parallel_for(policy,ff);
      ff.contribute();
    }
  } else {
    if (fpair->atom->ntypes > MAX_TYPES_STACKPARAMS) {
      PairComputeFunctor<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list);
      if (fpair->eflag || fpair->vflag) Kokkos::parallel_reduce(inum,ff,ev);
      else                              Kokkos::parallel_for(inum,ff);
      ff.contribute();
    } else {
      PairComputeFunctor<PairStyle,NEIGHFLAG,true,ZEROFLAG,Specialisation > ff(fpair,list);
      if (fpair->eflag || fpair->vflag) Kokkos::parallel_reduce(inum,ff,ev);
      else                              Kokkos::parallel_for(inum,ff);
      ff.contribute();
    }
  }
  return ev;
}

template<class PairStyle, class Specialisation = void>
EV_FLOAT pair_compute (PairStyle* fpair, NeighListKokkos<typename PairStyle::device_type>* list) {
  EV_FLOAT ev;
  if (fpair->neighflag == FULL) {
    if (utils::strmatch(fpair->lmp->force->pair_style,"^hybrid")) {
      fpair->fuse_force_clear_flag = 0;
      ev = pair_compute_neighlist<PairStyle,FULL,0,Specialisation> (fpair,list);
    } else {
      fpair->fuse_force_clear_flag = 1;
      ev = pair_compute_neighlist<PairStyle,FULL,1,Specialisation> (fpair,list);
    }
  } else if (fpair->neighflag == HALFTHREAD) {
    ev = pair_compute_neighlist<PairStyle,HALFTHREAD,0,Specialisation> (fpair,list);
  } else if (fpair->neighflag == HALF) {
    ev = pair_compute_neighlist<PairStyle,HALF,0,Specialisation> (fpair,list);
  }
  return ev;
}


#include "pair_kokkos_lj.h"
#include "pair_kokkos_lj_expr.h"
#include "pair_kokkos_coul.h"
#include "pair_kokkos_rhodo.h"
#include "pair_kokkos_rhodo_erfc.h"
#include "pair_kokkos_rhodo_erfc_expr.h"
#include "pair_kokkos_charmmfsw_coul_long.h"

template<class DeviceType>
struct PairVirialFDotRCompute {
  typedef DeviceType device_type;
  typedef ArrayTypes<DeviceType> AT;
  typedef EV_FLOAT value_type;
  typename AT::t_x_array_const_um x;
  typename AT::t_f_array_const_um f;
  const int offset;

  PairVirialFDotRCompute(  typename AT::t_x_array_const_um x_,
  typename AT::t_f_array_const_um f_,
  const int offset_):x(x_),f(f_),offset(offset_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int j, value_type &energy_virial) const {
    const int i = j + offset;
    energy_virial.v[0] += f(i,0)*x(i,0);
    energy_virial.v[1] += f(i,1)*x(i,1);
    energy_virial.v[2] += f(i,2)*x(i,2);
    energy_virial.v[3] += f(i,1)*x(i,0);
    energy_virial.v[4] += f(i,2)*x(i,0);
    energy_virial.v[5] += f(i,2)*x(i,1);
  }
};

template<class PairStyle>
void pair_virial_fdotr_compute(PairStyle* fpair) {
  EV_FLOAT virial;
  if (fpair->neighbor->includegroup == 0) {
    int nall = fpair->atom->nlocal + fpair->atom->nghost;
    Kokkos::parallel_reduce(nall,PairVirialFDotRCompute<typename PairStyle::device_type>(fpair->x,fpair->f,0),virial);
  } else {
    Kokkos::parallel_reduce(fpair->atom->nfirst,PairVirialFDotRCompute<typename PairStyle::device_type>(fpair->x,fpair->f,0),virial);
    EV_FLOAT virial_ghost;
    Kokkos::parallel_reduce(fpair->atom->nghost,PairVirialFDotRCompute<typename PairStyle::device_type>(fpair->x,fpair->f,fpair->atom->nlocal),virial_ghost);
    virial+=virial_ghost;
  }
  fpair->vflag_fdotr = 0;
  fpair->virial[0] = virial.v[0];
  fpair->virial[1] = virial.v[1];
  fpair->virial[2] = virial.v[2];
  fpair->virial[3] = virial.v[3];
  fpair->virial[4] = virial.v[4];
  fpair->virial[5] = virial.v[5];
}

}

#endif
#endif
