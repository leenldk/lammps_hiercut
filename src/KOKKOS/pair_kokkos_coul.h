namespace CoulKernels {

//  compute_item_custom(const int& ii,
//                       const NeighListKokkos<device_type> &list, const CoulTag& ) const {

template<class DeviceType, int EVFLAG>
__global__ void double_force_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, F_FLOAT cutsq, F_FLOAT cut_coulsq, F_FLOAT scale) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    const int itype = type(i);
    const F_FLOAT qtmp = q(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      //const F_FLOAT factor_lj = c.special_lj[0];
      //const F_FLOAT factor_coul = c.special_coul[0];
      const F_FLOAT factor_lj = 1.0;
      const F_FLOAT factor_coul = 1.0;
      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < cut_coulsq) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul;

          forcecoul = qqrd2e * scale * qtmp * q(j) * rinv;

          fpair += factor_coul*forcecoul*r2inv;
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < cut_coulsq) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = factor_coul * qqrd2e * scale * qtmp * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += fxtmp;
    f(i,1) += fytmp;
    f(i,2) += fztmp;
  }
}

template<class DeviceType, int EVFLAG>
__global__ void double_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x_rel, X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, F_FLOAT cutsq, F_FLOAT cut_coulsq, F_FLOAT scale) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x_rel(i,0);
    const X_FLOAT ytmp = x_rel(i,1);
    const X_FLOAT ztmp = x_rel(i,2);
    const int itype = type(i);
    const F_FLOAT qtmp = q(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const F_FLOAT factor_lj = 1.0;
      const F_FLOAT factor_coul = 1.0;
      int j = ni & DIRNEIGHMASK;
      const X_FLOAT delx = xtmp - x_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const X_FLOAT dely = ytmp - x_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const X_FLOAT delz = ztmp - x_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < cut_coulsq) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul;

          forcecoul = qqrd2e * scale * qtmp * q(j) * rinv;

          fpair += factor_coul*forcecoul*r2inv;
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < cut_coulsq) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = factor_coul * qqrd2e * scale * qtmp * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += fxtmp;
    f(i,1) += fytmp;
    f(i,2) += fztmp;
  }
}

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomDouble  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  // The copy of the pair style
  PairStyle c;
  typename AT::t_f_array f;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  using KKDeviceType = typename KKDevice<device_type>::value;
  using DUP = typename NeedDup<NEIGHFLAG,device_type>::value;

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

  PairComputeFunctorCustomDouble(PairStyle* c_ptr,
                          NeighListKokkos<device_type>* list_ptr, int ntotal):
  c(*c_ptr),list(*list_ptr) {
    // allocate duplicated memory
    f = c.f;
    d_eatom = c.d_eatom;
    d_vatom = c.d_vatom;
    dup_f     = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.f);
    dup_eatom = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.d_eatom);
    dup_vatom = Kokkos::Experimental::create_scatter_view<KKScatterSum, DUP>(c.d_vatom);
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctorCustomDouble() {c.copymode = 1; list.copymode = 1;};

  KOKKOS_INLINE_FUNCTION int sbmask(const int& j) const {
    return j >> SBBITS & 3;
  }

  void contribute() {
    Kokkos::Experimental::contribute(c.f, dup_f);

    if (c.eflag_atom)
      Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

    if (c.vflag_atom)
      Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
  }

  void contribute_custom() {
    //printf("perform contribute\n");
  }

  // Loop over neighbors of one atom with coulomb interaction
  // This function is called in parallel
  /*template<int EVFLAG, int NEWTON_PAIR>
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

        if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        }

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag) {
            if (rsq < (STACKPARAMS?c.m_cut_ljsq[itype][jtype]:c.d_cut_ljsq(itype,jtype))) {
              evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<c.nlocal)))?1.0:0.5)*evdwl;
            }
            if (rsq < (STACKPARAMS?c.m_cut_coulsq[itype][jtype]:c.d_cut_coulsq(itype,jtype))) {
              ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              ev.ecoul += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<c.nlocal)))?1.0:0.5)*ecoul;
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
  }*/

  template<int EVFLAG, int NEWTON_PAIR>
  KOKKOS_FUNCTION
  EV_FLOAT compute_item_custom(const int& ii,
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
      // const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      // const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
      // j &= NEIGHMASK;
      const F_FLOAT factor_lj = c.special_lj[0];
      const F_FLOAT factor_coul = c.special_coul[0];
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < c.m_cutsq[itype][jtype]) {

        F_FLOAT fpair = F_FLOAT();

        //if (rsq < c.m_cut_ljsq[itype][jtype])
        //  fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
        if (rsq < c.m_cut_coulsq[itype][jtype]) {
          //fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul;

          forcecoul = c.qqrd2e* c.m_params[itype][jtype].scale *
            qtmp *c.q(j) *rinv;

          fpair += factor_coul*forcecoul*r2inv;
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        // if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
        //   a_f(j,0) -= delx*fpair;
        //   a_f(j,1) -= dely*fpair;
        //   a_f(j,2) -= delz*fpair;
        // }

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag) {
            // if (rsq < c.m_cut_ljsq[itype][jtype]) {
            //   evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            //   ev.evdwl += (0.5)*evdwl;
            // }
            if (rsq < c.m_cut_coulsq[itype][jtype]) {
              // ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = factor_coul * c.qqrd2e * c.m_params[itype][jtype].scale
                * qtmp * c.q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          //if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl+ecoul,fpair,delx,dely,delz);
          if (c.vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            if (c.vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }

    a_f(i,0) += fxtmp;
    a_f(i,1) += fytmp;
    a_f(i,2) += fztmp;

    return ev;
  }

  KOKKOS_INLINE_FUNCTION
    void ev_tally(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    auto a_eatom = dup_eatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    auto a_vatom = dup_vatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    const int EFLAG = c.eflag;
    const int NEWTON_PAIR = c.newton_pair;
    const int VFLAG = c.vflag_either;

    // if (EFLAG) {
    //   if (c.eflag_atom) {
    //     const E_FLOAT epairhalf = 0.5 * epair;
    //     if (NEWTON_PAIR || i < c.nlocal) a_eatom[i] += epairhalf;
    //     if ((NEWTON_PAIR || j < c.nlocal) && NEIGHFLAG != FULL) a_eatom[j] += epairhalf;
    //   }
    // }

    if (VFLAG) {
      const E_FLOAT v0 = delx*delx*fpair;
      const E_FLOAT v1 = dely*dely*fpair;
      const E_FLOAT v2 = delz*delz*fpair;
      const E_FLOAT v3 = delx*dely*fpair;
      const E_FLOAT v4 = delx*delz*fpair;
      const E_FLOAT v5 = dely*delz*fpair;

      if (c.vflag_global) {
          ev.v[0] += 0.5*v0;
          ev.v[1] += 0.5*v1;
          ev.v[2] += 0.5*v2;
          ev.v[3] += 0.5*v3;
          ev.v[4] += 0.5*v4;
          ev.v[5] += 0.5*v5;
      }

      /*if (c.vflag_atom) {
        if (i < c.nlocal) {
          a_vatom(i,0) += 0.5*v0;
          a_vatom(i,1) += 0.5*v1;
          a_vatom(i,2) += 0.5*v2;
          a_vatom(i,3) += 0.5*v3;
          a_vatom(i,4) += 0.5*v4;
          a_vatom(i,5) += 0.5*v5;
        }
      }*/
    }
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    // if (c.newton_pair) compute_item_custom<0,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    // else 
    compute_item_custom<0,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &energy_virial) const {
    //if (c.newton_pair)
    //  energy_virial += compute_item_custom<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    //else
    energy_virial += compute_item_custom<1,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    if (USE_RELATIVE_COORD) {
      typename ArrayTypes<device_type>::t_x_array curr_x_rel = c.c_x_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_rel(i,0) = curr_x(i,0) - curr_bin_base(i,0);
        curr_x_rel(i,1) = curr_x(i,1) - curr_bin_base(i,1);
        curr_x_rel(i,2) = curr_x(i,2) - curr_bin_base(i,2);
      });
      Kokkos::fence();
    }

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      double_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.m_params[1][1].scale);
    }
    else {
      double_force_kernel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, c.q, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    if (USE_RELATIVE_COORD) {
      typename ArrayTypes<device_type>::t_x_array curr_x_rel = c.c_x_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_rel(i,0) = curr_x(i,0) - curr_bin_base(i,0);
        curr_x_rel(i,1) = curr_x(i,1) - curr_bin_base(i,1);
        curr_x_rel(i,2) = curr_x(i,2) - curr_bin_base(i,2);
      });
      Kokkos::fence();
    }

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      //printf("launch x_rel EV kernel\n");
      double_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.m_params[1][1].scale);
    }
    else {
      double_force_kernel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, c.q, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    return ev;
  }
};

template<class DeviceType, int EVFLAG>
__global__ void float_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel, float binsizex, float binsizey, float binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cutsq, float cut_coulsq, float scale) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const float xtmp = x_float_rel(i,0);
    const float ytmp = x_float_rel(i,1);
    const float ztmp = x_float_rel(i,2);
    const int itype = type(i);
    const float qtmp = q(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const float factor_lj = 1.0f;
      const float factor_coul = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx = xtmp - x_float_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely = ytmp - x_float_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz = ztmp - x_float_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq) {

        float fpair = 0.0f;

        if (rsq < cut_coulsq) {
          const float r2inv = 1.0/rsq;
          const float rinv = sqrt(r2inv);
          float forcecoul;

          forcecoul = qqrd2e * scale * qtmp * q(j) * rinv;

          fpair += factor_coul*forcecoul*r2inv;
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < cut_coulsq) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = factor_coul * qqrd2e * scale * qtmp * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += (F_FLOAT)fxtmp;
    f(i,1) += (F_FLOAT)fytmp;
    f(i,2) += (F_FLOAT)fztmp;
  }
}

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomFloat  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  // The copy of the pair style
  PairStyle c;
  typename AT::t_f_array f;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  NeighListKokkos<device_type> list;

  PairComputeFunctorCustomFloat(PairStyle* c_ptr,
                          NeighListKokkos<device_type>* list_ptr, int ntotal):
  c(*c_ptr),list(*list_ptr) {
    // allocate duplicated memory
    f = c.f;
    d_eatom = c.d_eatom;
    d_vatom = c.d_vatom;
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctorCustomFloat() {c.copymode = 1; list.copymode = 1;};

  KOKKOS_INLINE_FUNCTION int sbmask(const int& j) const {
    return j >> SBBITS & 3;
  }

  void contribute_custom() {}

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated || fpair -> ev_array.extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (USE_RELATIVE_COORD) {
      if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
        printf("lazy init x_float_rel\n"); fflush(stdout);
        fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float_rel = fpair -> x_float_rel;
        printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
      }
    }
    else {
      printf("float kernel not implemented for no RELATIVE_COORD\n");
      exit(-1);
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    if (USE_RELATIVE_COORD) {
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float_rel = c.x_float_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_float_rel(i,0) = (float)(curr_x(i,0) - curr_bin_base(i,0));
        curr_x_float_rel(i,1) = (float)(curr_x(i,1) - curr_bin_base(i,1));
        curr_x_float_rel(i,2) = (float)(curr_x(i,2) - curr_bin_base(i,2));
      });
      Kokkos::fence();
    }

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      float_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
          c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated || fpair -> ev_array.extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (USE_RELATIVE_COORD) {
      if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
        printf("lazy init x_float_rel\n"); fflush(stdout);
        fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float_rel = fpair -> x_float_rel;
        printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
      }
    }
    else {
      printf("float kernel not implemented for no RELATIVE_COORD\n");
      exit(-1);
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    if (USE_RELATIVE_COORD) {
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float_rel = c.x_float_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_float_rel(i,0) = (float)(curr_x(i,0) - curr_bin_base(i,0));
        curr_x_float_rel(i,1) = (float)(curr_x(i,1) - curr_bin_base(i,1));
        curr_x_float_rel(i,2) = (float)(curr_x(i,2) - curr_bin_base(i,2));
      });
      Kokkos::fence();
    }

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      float_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
          c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    return ev;
  }
};

template <class DeviceType>
__global__ void init_aos_xhalf_rel_kernel(int ntotal, AoS_half* x_half_rel, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type, 
  typename ArrayTypes<DeviceType>::t_x_array_randomread x,
  typename ArrayTypes<DeviceType>::t_x_array_randomread bin_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ntotal) {
    x_half_rel[i].x[0] = __double2half(x(i,0) - bin_base(i,0));
    x_half_rel[i].x[1] = __double2half(x(i,1) - bin_base(i,1));
    x_half_rel[i].x[2] = __double2half(x(i,2) - bin_base(i,2));
    x_half_rel[i].type = static_cast<short>(type(i));
  }
}

template<class DeviceType, int EVFLAG>
__global__ void half_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, __half binsizex_h, __half binsizey_h, __half binsizez_h, 
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cutsq, float cut_coulsq, float scale) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const AoS_half half_data = x_half_rel[i];
    const __half xtmp = half_data.x[0];
    const __half ytmp = half_data.x[1];
    const __half ztmp = half_data.x[2];
    const int itype = half_data.type;
    const __half qtmp = __double2half(q(i));

    __half zero_h = __ushort_as_half((unsigned short)0x0000U);
    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    __half fxtmp = zero_h;
    __half fytmp = zero_h;
    __half fztmp = zero_h;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const __half factor_lj = one_h;
      const __half factor_coul = one_h;
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx = xtmp - half_data_j.x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex_h;
      const __half dely = ytmp - half_data_j.x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey_h;
      const __half delz = ztmp - half_data_j.x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez_h;

      const int jtype = half_data_j.type;
      const __half rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < __float2half(cutsq)) {

        __half fpair = zero_h;

        if (rsq < __float2half(cut_coulsq)) {
          const __half r2inv = one_h/rsq;
          const __half rinv = hsqrt(r2inv);
          __half forcecoul;

          forcecoul = __float2half(qqrd2e) * __float2half(scale) * qtmp * __double2half(q(j)) * rinv;

          fpair += factor_coul*forcecoul*r2inv;
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < __float2half(cut_coulsq)) {
              const F_FLOAT r2inv = 1.0/(double)(__half2float(rsq));
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = __half2float(factor_coul) * qqrd2e * scale * __half2float(qtmp) * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += __half2float(fxtmp);
    f(i,1) += __half2float(fytmp);
    f(i,2) += __half2float(fztmp);
  }
}

// simulate using table for query
template<class DeviceType, int EVFLAG>
__global__ void half_force_kernel_x_rel_sim_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, __half binsizex_h, __half binsizey_h, __half binsizez_h, 
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cutsq, float cut_coulsq, float scale) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const AoS_half half_data = x_half_rel[i];
    const __half xtmp = half_data.x[0];
    const __half ytmp = half_data.x[1];
    const __half ztmp = half_data.x[2];
    const int itype = half_data.type;
    const __half qtmp = __double2half(q(i));

    __half zero_h = __ushort_as_half((unsigned short)0x0000U);
    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    __half fxtmp = zero_h;
    __half fytmp = zero_h;
    __half fztmp = zero_h;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const __half factor_lj = one_h;
      const __half factor_coul = one_h;
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx = xtmp - half_data_j.x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex_h;
      const __half dely = ytmp - half_data_j.x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey_h;
      const __half delz = ztmp - half_data_j.x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez_h;

      const int jtype = half_data_j.type;
      const __half rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < __float2half(cutsq)) {

        __half fpair = zero_h;

        if (rsq < __float2half(cut_coulsq)) {
          double rsq_d = (double)(__half2float(rsq));
          const double r2inv_d = 1.0 / rsq_d;
          const double rinv_d = sqrt(r2inv_d);
          double mid_res_d = (double)qqrd2e * (double)scale * rinv_d * r2inv_d;
          __half mid_res_h = __double2half(mid_res_d);
          //const __half r2inv = one_h/rsq;
          //const __half rinv = hsqrt(r2inv);
          //__half forcecoul;

          //forcecoul = __float2half(qqrd2e) * __float2half(scale) * qtmp * __double2half(q(j)) * rinv;

          //fpair += factor_coul*__float2half(qqrd2e) * __float2half(scale) * rinv*r2inv * qtmp * __double2half(q(j));
          fpair += factor_coul*mid_res_h * qtmp * __double2half(q(j));
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < __float2half(cut_coulsq)) {
              const F_FLOAT r2inv = 1.0/(double)(__half2float(rsq));
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = __half2float(factor_coul) * qqrd2e * scale * __half2float(qtmp) * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = delx*delx*fpair;
            const E_FLOAT v1 = dely*dely*fpair;
            const E_FLOAT v2 = delz*delz*fpair;
            const E_FLOAT v3 = delx*dely*fpair;
            const E_FLOAT v4 = delx*delz*fpair;
            const E_FLOAT v5 = dely*delz*fpair;

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += __half2float(fxtmp);
    f(i,1) += __half2float(fytmp);
    f(i,2) += __half2float(fztmp);
  }
}


template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomHalf  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  // The copy of the pair style
  PairStyle c;
  typename AT::t_f_array f;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  NeighListKokkos<device_type> list;

  PairComputeFunctorCustomHalf(PairStyle* c_ptr,
                          NeighListKokkos<device_type>* list_ptr, int ntotal):
  c(*c_ptr),list(*list_ptr) {
    // allocate duplicated memory
    f = c.f;
    d_eatom = c.d_eatom;
    d_vatom = c.d_vatom;
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctorCustomHalf() {c.copymode = 1; list.copymode = 1;};

  KOKKOS_INLINE_FUNCTION int sbmask(const int& j) const {
    return j >> SBBITS & 3;
  }

  void contribute_custom() {}

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated || fpair -> ev_array.extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (USE_RELATIVE_COORD) {
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half_rel\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_rel);
        }
        cudaMalloc((void**)&(fpair -> x_half_rel), (fpair -> x).extent(0) * sizeof(AoS_half));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_rel = fpair -> x_half_rel;
        printf("x_half_rel extent : %d\n", fpair -> x_half_size);
      }
    }
    else {
      printf("half kernel not implemented for no RELATIVE_COORD\n");
      exit(-1);
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      init_aos_xhalf_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel, c.type, c.x, c.bin_base);
    }
    cudaDeviceSynchronize();

    threadsPerBlock = 128;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      half_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
          c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
      // half_force_kernel_x_rel_sim_table<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
      //     c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated || fpair -> ev_array.extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (USE_RELATIVE_COORD) {
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half_rel\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_rel);
        }
        cudaMalloc((void**)&(fpair -> x_half_rel), (fpair -> x).extent(0) * sizeof(AoS_half));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_rel = fpair -> x_half_rel;
        printf("x_half_rel extent : %d\n", fpair -> x_half_size);
      }
    }
    else {
      printf("float kernel not implemented for no RELATIVE_COORD\n");
      exit(-1);
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      init_aos_xhalf_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel, c.type, c.x, c.bin_base);
    }
    cudaDeviceSynchronize();

    threadsPerBlock = 128;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      half_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
          c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
      // half_force_kernel_x_rel_sim_table<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
      //     c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    return ev;
  }
};

template <class DeviceType>
__global__ void init_aos_xfhmix_rel_kernel(int ntotal, AoS_half* x_half_rel, 
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type, 
  typename ArrayTypes<DeviceType>::t_x_array_randomread x,
  typename ArrayTypes<DeviceType>::t_x_array_randomread bin_base) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ntotal) {
    double xtmp = x(i,0) - bin_base(i,0);
    double ytmp = x(i,1) - bin_base(i,1);
    double ztmp = x(i,2) - bin_base(i,2);
    x_float_rel(i,0) = (float)xtmp;
    x_float_rel(i,1) = (float)ytmp;
    x_float_rel(i,2) = (float)ztmp;
    x_half_rel[i].x[0] = __double2half(xtmp);
    x_half_rel[i].x[1] = __double2half(ytmp);
    x_half_rel[i].x[2] = __double2half(ztmp);
    x_half_rel[i].type = static_cast<short>(type(i));
  }
}

template<class DeviceType, int EVFLAG>
__global__ void fhmix_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread fhcut_split,
  const float binsizex, const float binsizey, const float binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cutsq, float cut_coulsq, float scale) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i,0);
    const float ytmp_f = x_float_rel(i,1);
    const float ztmp_f = x_float_rel(i,2);
    const int itype = type(i);
    const float qtmp_f = q(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int fh_cut = fhcut_split(i);
    //const int jnum = d_numneigh(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    int jj;
    for (jj = 0; jj < fh_cut; jj++) {
      int ni = neighbors_i(jj);
      const float factor_lj_f = 1.0f;
      const float factor_coul_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {

        float fpair_f = 0.0f;

        if (rsq_f < cut_coulsq) {
          const float r2inv_f = 1.0f/rsq_f;
          const float rinv_f = sqrt(r2inv_f);
          float forcecoul_f;

          forcecoul_f = qqrd2e * scale * qtmp_f * (float)q(j) * rinv_f;

          fpair_f += factor_coul_f * forcecoul_f * r2inv_f;
        }

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq_f < cut_coulsq) {
              const F_FLOAT r2inv = 1.0/rsq_f;
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = factor_coul_f * qqrd2e * scale * qtmp_f * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = delx_f * delx_f * fpair_f;
            const E_FLOAT v1 = dely_f * dely_f * fpair_f;
            const E_FLOAT v2 = delz_f * delz_f * fpair_f;
            const E_FLOAT v3 = delx_f * dely_f * fpair_f;
            const E_FLOAT v4 = delx_f * delz_f * fpair_f;
            const E_FLOAT v5 = dely_f * delz_f * fpair_f;

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    const __half xtmp_h = __float2half(xtmp_f);
    const __half ytmp_h = __float2half(ytmp_f);
    const __half ztmp_h = __float2half(ztmp_f);
    const __half qtmp_h = __float2half(qtmp_f);

    __half zero_h = __ushort_as_half((unsigned short)0x0000U);
    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    const int jnum = d_numneigh(i);

    __half fxtmp_h = zero_h;
    __half fytmp_h = zero_h;
    __half fztmp_h = zero_h;

    for (; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const __half factor_lj_h = one_h;
      const __half factor_coul_h = one_h;
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx_h = xtmp_h - half_data_j.x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * __float2half(binsizex);
      const __half dely_h = ytmp_h - half_data_j.x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * __float2half(binsizey);
      const __half delz_h = ztmp_h - half_data_j.x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * __float2half(binsizez);

      const int jtype = half_data_j.type;
      const __half rsq_h = delx_h * delx_h + dely_h * dely_h + delz_h * delz_h;

      if (rsq_h < __float2half(cutsq)) {

        __half fpair_h = zero_h;

        if (rsq_h < __float2half(cut_coulsq)) {
          const __half r2inv_h = one_h / rsq_h;
          const __half rinv_h = hsqrt(r2inv_h);
          __half forcecoul_h;

          forcecoul_h = __float2half(qqrd2e) * __float2half(scale) * qtmp_h * __double2half(q(j)) * rinv_h;

          fpair_h += factor_coul_h * forcecoul_h * r2inv_h;
        }

        fxtmp_h += delx_h * fpair_h;
        fytmp_h += dely_h * fpair_h;
        fztmp_h += delz_h * fpair_h;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq_h < __float2half(cut_coulsq)) {
              const F_FLOAT r2inv = 1.0/(double)(__half2float(rsq_h));
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = __half2float(factor_coul_h) * qqrd2e * scale * __half2float(qtmp_h) * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h * delx_h) * __half2float(fpair_h);
            const E_FLOAT v1 = __half2float(dely_h * dely_h) * __half2float(fpair_h);
            const E_FLOAT v2 = __half2float(delz_h * delz_h) * __half2float(fpair_h);
            const E_FLOAT v3 = __half2float(delx_h * dely_h) * __half2float(fpair_h);
            const E_FLOAT v4 = __half2float(delx_h * delz_h) * __half2float(fpair_h);
            const E_FLOAT v5 = __half2float(dely_h * delz_h) * __half2float(fpair_h);

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += __half2float(fxtmp_h);
    f(i,1) += __half2float(fytmp_h);
    f(i,2) += __half2float(fztmp_h);
  }
}

template<class DeviceType, int EVFLAG>
__global__ void fhmix_force_kernel_x_rel_sim_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread fhcut_split,
  const float binsizex, const float binsizey, const float binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cutsq, float cut_coulsq, float scale) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i,0);
    const float ytmp_f = x_float_rel(i,1);
    const float ztmp_f = x_float_rel(i,2);
    const int itype = type(i);
    const float qtmp_f = q(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int fh_cut = fhcut_split(i);
    //const int jnum = d_numneigh(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    int jj;
    for (jj = 0; jj < fh_cut; jj++) {
      int ni = neighbors_i(jj);
      const float factor_lj_f = 1.0f;
      const float factor_coul_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {

        float fpair_f = 0.0f;

        if (rsq_f < cut_coulsq) {
          const float r2inv_f = 1.0f/rsq_f;
          const float rinv_f = sqrt(r2inv_f);
          float forcecoul_f;

          forcecoul_f = qqrd2e * scale * qtmp_f * (float)q(j) * rinv_f;

          fpair_f += factor_coul_f * forcecoul_f * r2inv_f;
        }

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq_f < cut_coulsq) {
              const F_FLOAT r2inv = 1.0/rsq_f;
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = factor_coul_f * qqrd2e * scale * qtmp_f * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = delx_f * delx_f * fpair_f;
            const E_FLOAT v1 = dely_f * dely_f * fpair_f;
            const E_FLOAT v2 = delz_f * delz_f * fpair_f;
            const E_FLOAT v3 = delx_f * dely_f * fpair_f;
            const E_FLOAT v4 = delx_f * delz_f * fpair_f;
            const E_FLOAT v5 = dely_f * delz_f * fpair_f;

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    const __half xtmp_h = __float2half(xtmp_f);
    const __half ytmp_h = __float2half(ytmp_f);
    const __half ztmp_h = __float2half(ztmp_f);
    const __half qtmp_h = __float2half(qtmp_f);

    __half zero_h = __ushort_as_half((unsigned short)0x0000U);
    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    const int jnum = d_numneigh(i);

    __half fxtmp_h = zero_h;
    __half fytmp_h = zero_h;
    __half fztmp_h = zero_h;

    for (; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const __half factor_lj_h = one_h;
      const __half factor_coul_h = one_h;
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx_h = xtmp_h - half_data_j.x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * __float2half(binsizex);
      const __half dely_h = ytmp_h - half_data_j.x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * __float2half(binsizey);
      const __half delz_h = ztmp_h - half_data_j.x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * __float2half(binsizez);

      const int jtype = half_data_j.type;
      const __half rsq_h = delx_h * delx_h + dely_h * dely_h + delz_h * delz_h;

      if (rsq_h < __float2half(cutsq)) {

        __half fpair_h = zero_h;

        if (rsq_h < __float2half(cut_coulsq)) {
          //const __half r2inv_h = one_h / rsq_h;
          //const __half rinv_h = hsqrt(r2inv_h);
          //__half forcecoul_h;
          double rsq_d = (double)(__half2float(rsq_h));
          const double r2inv_d = 1.0 / rsq_d;
          const double rinv_d = sqrt(r2inv_d);
          double mid_res_d = (double)qqrd2e * (double)scale * rinv_d * r2inv_d;
          __half mid_res_h = __double2half(mid_res_d);

          //forcecoul_h = __float2half(qqrd2e) * __float2half(scale) * qtmp_h * __double2half(q(j)) * rinv_h;

          //fpair_h += factor_coul_h * forcecoul_h * r2inv_h;
          //fpair_h += factor_coul_h * float2half(qqrd2e) * __float2half(scale) * rinv_h * r2inv_h * qtmp_h * __double2half(q(j));
          fpair_h += factor_coul_h * mid_res_h * qtmp_h * __double2half(q(j));
        }

        fxtmp_h += delx_h * fpair_h;
        fytmp_h += dely_h * fpair_h;
        fztmp_h += delz_h * fpair_h;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq_h < __float2half(cut_coulsq)) {
              const F_FLOAT r2inv = 1.0/(double)(__half2float(rsq_h));
              const F_FLOAT rinv = sqrt(r2inv);

              ecoul = __half2float(factor_coul_h) * qqrd2e * scale * __half2float(qtmp_h) * q(j) * rinv;

              ev.ecoul += 0.5*ecoul;
            }
          }

          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h * delx_h) * __half2float(fpair_h);
            const E_FLOAT v1 = __half2float(dely_h * dely_h) * __half2float(fpair_h);
            const E_FLOAT v2 = __half2float(delz_h * delz_h) * __half2float(fpair_h);
            const E_FLOAT v3 = __half2float(delx_h * dely_h) * __half2float(fpair_h);
            const E_FLOAT v4 = __half2float(delx_h * delz_h) * __half2float(fpair_h);
            const E_FLOAT v5 = __half2float(dely_h * delz_h) * __half2float(fpair_h);

            if (vflag_global) {
              ev.v[0] += 0.5*v0;
              ev.v[1] += 0.5*v1;
              ev.v[2] += 0.5*v2;
              ev.v[3] += 0.5*v3;
              ev.v[4] += 0.5*v4;
              ev.v[5] += 0.5*v5;
            }        
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    f(i,0) += __half2float(fxtmp_h);
    f(i,1) += __half2float(fytmp_h);
    f(i,2) += __half2float(fztmp_h);
  }
}

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomFhmix  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  // The copy of the pair style
  PairStyle c;
  typename AT::t_f_array f;
  typename AT::t_efloat_1d d_eatom;
  typename AT::t_virial_array d_vatom;

  NeighListKokkos<device_type> list;

  PairComputeFunctorCustomFhmix(PairStyle* c_ptr,
                          NeighListKokkos<device_type>* list_ptr, int ntotal):
  c(*c_ptr),list(*list_ptr) {
    // allocate duplicated memory
    f = c.f;
    d_eatom = c.d_eatom;
    d_vatom = c.d_vatom;
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctorCustomFhmix() {c.copymode = 1; list.copymode = 1;};

  KOKKOS_INLINE_FUNCTION int sbmask(const int& j) const {
    return j >> SBBITS & 3;
  }

  void contribute_custom() {}

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated || fpair -> ev_array.extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (USE_RELATIVE_COORD) {
      if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
        printf("lazy init x_float_rel\n"); fflush(stdout);
        fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float_rel = fpair -> x_float_rel;
        printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
      }
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half_rel\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_rel);
        }
        cudaMalloc((void**)&(fpair -> x_half_rel), (fpair -> x).extent(0) * sizeof(AoS_half));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_rel = fpair -> x_half_rel;
        printf("x_half_rel extent : %d\n", fpair -> x_half_size);
      }
    }
    else {
      printf("fhmix kernel not implemented for no RELATIVE_COORD\n");
      exit(-1);
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      init_aos_xfhmix_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel, c.x_float_rel, c.type, c.x, c.bin_base);
    }
    cudaDeviceSynchronize();

    threadsPerBlock = 128;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      fhmix_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.x_float_rel,
          c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, 
          c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
      // fhmix_force_kernel_x_rel_sim_table<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.x_float_rel,
      //     c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, 
      //     c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated || fpair -> ev_array.extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (USE_RELATIVE_COORD) {
      if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
        printf("lazy init x_float_rel\n"); fflush(stdout);
        fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float_rel = fpair -> x_float_rel;
        printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
      }
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half_rel\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_rel);
        }
        cudaMalloc((void**)&(fpair -> x_half_rel), (fpair -> x).extent(0) * sizeof(AoS_half));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_rel = fpair -> x_half_rel;
        printf("x_half_rel extent : %d\n", fpair -> x_half_size);
      }
    }
    else {
      printf("float kernel not implemented for no RELATIVE_COORD\n");
      exit(-1);
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      init_aos_xfhmix_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel, c.x_float_rel, c.type, c.x, c.bin_base);
    }
    cudaDeviceSynchronize();

    threadsPerBlock = 128;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      fhmix_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.x_float_rel,
          c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, 
          c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
      // fhmix_force_kernel_x_rel_sim_table<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.x_float_rel,
      //     c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, 
      //     c.type, c.q, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.m_cutsq[1][1], (float)c.m_cut_coulsq[1][1], (float)c.m_params[1][1].scale);
    }

    cudaDeviceSynchronize();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    return ev;
  }
};


template<class PairStyle, PRECISION_TYPE PRECTYPE, unsigned NEIGHFLAG, int USE_RELATIVE_COORD, int ZEROFLAG = 0, class Specialisation = void>
EV_FLOAT pair_compute_neighlist_custom (PairStyle* fpair, typename std::enable_if<(NEIGHFLAG&PairStyle::EnabledNeighFlags) != 0, NeighListKokkos<typename PairStyle::device_type>*>::type list) {

  //printf("in pair_compute_neighlist_custom\n");

  if(NEIGHFLAG != FULL) {
    printf("ERROR: NEIGHFLAG is not FULL\n");
    exit(1);
  }
  if(ZEROFLAG != 1) {
    printf("ERROR: ZEROFLAG is not 1\n");
    exit(1);
  }
  if(!std::is_same<Specialisation, void>::value) {
    printf("ERROR: Specialisation is not void\n");
    exit(1);
  }

  EV_FLOAT ev;
  if (!fpair->lmp->kokkos->neigh_thread_set)
    if (list->inum <= 16384 && NEIGHFLAG == FULL)
      fpair->lmp->kokkos->neigh_thread = 1;

  if(fpair->lmp->kokkos->neigh_thread != 0) {
    printf("ERROR: NEIGH_THREAD is not zero\n");
    exit(1);
  }

  if (fpair->atom->ntypes > MAX_TYPES_STACKPARAMS) {
    printf("ERROR: atom->ntypes is greater than MAX_TYPES_STACKPARAMS\n");
    exit(1);
  }

  if (!std::is_same<typename DoCoul<PairStyle::COUL_FLAG>::type, CoulTag>::value) {
    printf("ERROR: DoCoul<PairStyle::COUL_FLAG>::type is not CoulTag\n");
    exit(1);
  }

  if(PRECTYPE == DOUBLE_PREC) {
    PairComputeFunctorCustomDouble<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD, ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    //PairComputeFunctor<PairStyle,NEIGHFLAG,true,ZEROFLAG,Specialisation > ff(fpair,list);
    if (fpair->eflag || fpair->vflag) {
      //Kokkos::parallel_reduce(list->inum,ff,ev);
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      //Kokkos::parallel_for(list->inum,ff);
      //ff.test_kernel_launch(list->inum);
      ff.kernel_launch(list->inum, fpair);
    }
    //ff.contribute();
    ff.contribute_custom();
  }
  else if(PRECTYPE == FLOAT_PREC) {
    // printf("call Coul float kernel\n");
    PairComputeFunctorCustomFloat<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute_custom();
  }
  else if(PRECTYPE == HALF_PREC) {
    // printf("call Coul half kernel\n");
    PairComputeFunctorCustomHalf<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute_custom();
  }
  else if(PRECTYPE == HFMIX_PREC) {
    //printf("call Coul hfmix kernel\n");
    PairComputeFunctorCustomFhmix<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute_custom();
  }
  else {
    printf("ERROR: PRECTYPE not implemented\n");
    exit(1);
  }

  return ev;
}

template<class PairStyle, PRECISION_TYPE PRECTYPE, class Specialisation = void>
EV_FLOAT pair_compute_custom (PairStyle* fpair, NeighListKokkos<typename PairStyle::device_type>* list) {
  EV_FLOAT ev;
  fpair->fuse_force_clear_flag = 1;
  if(fpair->use_relative_coord) {
    ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 1, 1, Specialisation> (fpair,list);
  }
  else {
    ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 0, 1, Specialisation> (fpair,list);
  }
  return ev;
}

} // namespace CoulKernels
