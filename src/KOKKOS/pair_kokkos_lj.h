namespace LJKernels {
/*__device__ void ev_tally_double(EV_FLOAT &ev, const F_FLOAT &fpair, const F_FLOAT &delx,
                const F_FLOAT &dely, const F_FLOAT &delz, int vflag_either, int vflag_global)
{
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
*/

template <class DeviceType>
__global__ void double_force_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    const int itype = type(i);

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
      //const F_FLOAT factor_lj = c.special_lj[j >> SBBITS & 3];
      const F_FLOAT factor_lj = 1.0;
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      // if (rsq < c.m_cutsq[itype][jtype]) {
      //   //const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
      //   const F_FLOAT r2inv = 1.0/rsq;
      //   const F_FLOAT r6inv = r2inv*r2inv*r2inv;

      //   const F_FLOAT forcelj = r6inv *
      //     (c.m_params[itype][jtype].lj1*r6inv - c.m_params[itype][jtype].lj2);

      //   const F_FLOAT fpair = factor_lj*forcelj*r2inv;

      //   fxtmp += delx*fpair;
      //   fytmp += dely*fpair;
      //   fztmp += delz*fpair;
      // }
      if (rsq < cutsq) {
        //const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
        const F_FLOAT r2inv = 1.0/rsq;
        const F_FLOAT r6inv = r2inv*r2inv*r2inv;

        const F_FLOAT forcelj = r6inv *
          (48*r6inv - 24);

        const F_FLOAT fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;
      }
    }

    f(i,0) += fxtmp;
    f(i,1) += fytmp;
    f(i,2) += fztmp;
  }
}

template <class DeviceType, int EVFLAG>
__global__ void double_force_kernel_xdata(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const X_FLOAT* x_data = x.data();
    const X_FLOAT xtmp = x_data[3 * i + 0];
    const X_FLOAT ytmp = x_data[3 * i + 1];
    const X_FLOAT ztmp = x_data[3 * i + 2];
    const int itype = type(i);

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
      //const F_FLOAT factor_lj = c.special_lj[j >> SBBITS & 3];
      const F_FLOAT factor_lj = 1.0;
      j &= NEIGHMASK;
      //const X_FLOAT delx = xtmp - x(j,0);
      //const X_FLOAT dely = ytmp - x(j,1);
      //const X_FLOAT delz = ztmp - x(j,2);
      const X_FLOAT delx = xtmp - x_data[3 * j + 0];
      const X_FLOAT dely = ytmp - x_data[3 * j + 1];
      const X_FLOAT delz = ztmp - x_data[3 * j + 2];
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      // if (rsq < c.m_cutsq[itype][jtype]) {
      //   //const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
      //   const F_FLOAT r2inv = 1.0/rsq;
      //   const F_FLOAT r6inv = r2inv*r2inv*r2inv;

      //   const F_FLOAT forcelj = r6inv *
      //     (c.m_params[itype][jtype].lj1*r6inv - c.m_params[itype][jtype].lj2);

      //   const F_FLOAT fpair = factor_lj*forcelj*r2inv;

      //   fxtmp += delx*fpair;
      //   fytmp += dely*fpair;
      //   fztmp += delz*fpair;
      // }
      if (rsq < cutsq) {
        //const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
        const F_FLOAT r2inv = 1.0/rsq;
        const F_FLOAT r6inv = r2inv*r2inv*r2inv;

        const F_FLOAT forcelj = r6inv *
          (48*r6inv - 24);

        const F_FLOAT fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            //evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            //ev.evdwl += 0.5*evdwl;
            //const F_FLOAT r2inv = 1.0/rsq;
            //const F_FLOAT r6inv = r2inv*r2inv*r2inv;

            evdwl = r6inv*(4.0 * r6inv -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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

template <class DeviceType, int EVFLAG>
__global__ void double_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x_rel, X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x_rel(i,0);
    const X_FLOAT ytmp = x_rel(i,1);
    const X_FLOAT ztmp = x_rel(i,2);
    const int itype = type(i);

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
      //const F_FLOAT factor_lj = c.special_lj[j >> SBBITS & 3];
      const F_FLOAT factor_lj = 1.0;
      int j = ni & DIRNEIGHMASK;
      const X_FLOAT delx = xtmp - x_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const X_FLOAT dely = ytmp - x_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const X_FLOAT delz = ztmp - x_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      // if (rsq < c.m_cutsq[itype][jtype]) {
      //   //const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
      //   const F_FLOAT r2inv = 1.0/rsq;
      //   const F_FLOAT r6inv = r2inv*r2inv*r2inv;

      //   const F_FLOAT forcelj = r6inv *
      //     (c.m_params[itype][jtype].lj1*r6inv - c.m_params[itype][jtype].lj2);

      //   const F_FLOAT fpair = factor_lj*forcelj*r2inv;

      //   fxtmp += delx*fpair;
      //   fytmp += dely*fpair;
      //   fztmp += delz*fpair;
      // }
      if (rsq < cutsq) {
        //const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
        const F_FLOAT r2inv = 1.0/rsq;
        const F_FLOAT r6inv = r2inv*r2inv*r2inv;

        const F_FLOAT forcelj = r6inv *
          (48*r6inv - 24);

        const F_FLOAT fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            //evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            //ev.evdwl += 0.5*evdwl;
            //const F_FLOAT r2inv = 1.0/rsq;
            //const F_FLOAT r6inv = r2inv*r2inv*r2inv;

            evdwl = r6inv*(4.0 * r6inv -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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


template <class DeviceType, int EVFLAG>
__global__ void double_kernel_x_rel_check(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x_rel, typename ArrayTypes<DeviceType>::t_x_array_randomread x,
  X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x_rel(i,0);
    const X_FLOAT ytmp = x_rel(i,1);
    const X_FLOAT ztmp = x_rel(i,2);

    const X_FLOAT xtmp_org = x(i,0);
    const X_FLOAT ytmp_org = x(i,1);
    const X_FLOAT ztmp_org = x(i,2);

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    if(i == 0) {
      printf("binsize : %f %f %f\n", binsizex, binsizey, binsizez);
    }

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      //const F_FLOAT factor_lj = c.special_lj[j >> SBBITS & 3];
      const F_FLOAT factor_lj = 1.0;
      int j = ni & DIRNEIGHMASK;
      const X_FLOAT delx = xtmp - x_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const X_FLOAT dely = ytmp - x_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const X_FLOAT delz = ztmp - x_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;

      const X_FLOAT delx_org = xtmp_org - x(j,0);
      const X_FLOAT dely_org = ytmp_org - x(j,1);
      const X_FLOAT delz_org = ztmp_org - x(j,2);

      if(i == 0) {
        printf("%d %d %f %f %f %f, del : %d, %d, %d\n", i, j, xtmp, x_rel(j,0), xtmp_org, x(j,0), 
          ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
          ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT),
          ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT));
        if(fabs(delx - delx_org) > 1e-9 || fabs(dely - dely_org) > 1e-9 || fabs(delz - delz_org) > 1e-9) {
          printf("ERROR: x_rel mismatch %i %i %f %f %f %f %f %f\n",i,j,delx,delx_org,dely,dely_org,delz,delz_org);
        }
      }
    }
  }
}

template <class PairStyle>
//__global__ void test_kernel(int ntotal, const NeighListKokkos<typename PairStyle::device_type> list, 
//  PairStyle c, typename ArrayTypes<typename PairStyle::device_type>::t_f_array f, double* f_ptr) {
__global__ void test_kernel(int ntotal, typename ArrayTypes<typename PairStyle::device_type>::t_f_array f, double* f_ptr) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    //const int i = list.d_ilist[ii];
    const int i = ii;
    //f(i,0) += c.x(i,0) * 1e-10;
    //f(i,1) += c.x(i,1) * 1e-10;
    //f(i,2) += c.x(i,2) * 1e-10;
    //f(i,0) += 1e-10;
    //f(i,1) += 1e-10;
    //f(i,2) += 1e-10;
    f_ptr[i] = i;
  }
}


template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, NEIGH_SEP_STRATEGY NEIGH_STG = NO_NEIGH_SEP, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomDouble  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  //Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array;

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
    Kokkos::Timer init_timer;
    init_timer.reset();
    //ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
    c_ptr->init_time += init_timer.seconds();
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctorCustomDouble() {c.copymode = 1; list.copymode = 1;};

  KOKKOS_INLINE_FUNCTION int sbmask(const int& j) const {
    return j >> SBBITS & 3;
  }

  /*void contribute() {
    Kokkos::Experimental::contribute(c.f, dup_f);

    if (c.eflag_atom)
      Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

    if (c.vflag_atom)
      Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
  }*/

  void contribute_custom() {
    //Kokkos::Experimental::contribute(c.f, dup_f);

    /*if (c.eflag_atom)
      Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

    if (c.vflag_atom)
      Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
    */
  }

  // Loop over neighbors of one atom without coulomb interaction
  // This function is called in parallel
  /*template<int EVFLAG, int NEWTON_PAIR>
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

        if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        }

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (c.eflag) {
            evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            ev.evdwl += (((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD)&&(NEWTON_PAIR||(j<c.nlocal)))?1.0:0.5)*evdwl;
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
        if (NEIGHFLAG!=FULL) {
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
  */

  // NEWTON_PAIR = 0, NEIGHFLAG = FULL, ZEROFLAG = 1, STACKPARAMS = 1, c.eflag_atom = 0, c.vflag_atom = 0
  template<int EVFLAG>
  KOKKOS_FUNCTION
  EV_FLOAT compute_item_custom(const int& ii,
                        const NeighListKokkos<device_type> &list, const NoCoulTag&) const {

    //auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

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

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < c.m_cutsq[itype][jtype]) {

        const F_FLOAT fpair = factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (c.eflag) {
            evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
            ev.evdwl += 0.5*evdwl;
          }

          if (c.vflag_either || c.eflag_atom) ev_tally_custom(ev,i,j,evdwl,fpair,delx,dely,delz);
        }
      }

    }

    f(i,0) += fxtmp;
    f(i,1) += fytmp;
    f(i,2) += fztmp;

    return ev;
  }

  KOKKOS_INLINE_FUNCTION
    void ev_tally_custom(EV_FLOAT &ev, const int &i, const int &j,
      const F_FLOAT &epair, const F_FLOAT &fpair, const F_FLOAT &delx,
                  const F_FLOAT &dely, const F_FLOAT &delz) const
  {
    //auto a_eatom = dup_eatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
    //auto a_vatom = dup_vatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

    const int EFLAG = c.eflag;
    const int NEWTON_PAIR = c.newton_pair;
    const int VFLAG = c.vflag_either;

    /*if (EFLAG) {
      if (c.eflag_atom) {
        const E_FLOAT epairhalf = 0.5 * epair;
        if (NEWTON_PAIR || i < c.nlocal) a_eatom[i] += epairhalf;
        if ((NEWTON_PAIR || j < c.nlocal) && NEIGHFLAG != FULL) a_eatom[j] += epairhalf;
      }
    }*/

    if (VFLAG) {
      const E_FLOAT v0 = delx*delx*fpair;
      const E_FLOAT v1 = dely*dely*fpair;
      const E_FLOAT v2 = delz*delz*fpair;
      const E_FLOAT v3 = delx*dely*fpair;
      const E_FLOAT v4 = delx*delz*fpair;
      const E_FLOAT v5 = dely*delz*fpair;

      if (c.vflag_global) {
        if (NEIGHFLAG!=FULL) {
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

      /*if (c.vflag_atom) {
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
      }*/
    }
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

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

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      double_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }
    else {
      //double_force_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
      //    ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, f, (float)c.m_cutsq[1][1]);
      double_force_kernel_xdata<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }
    cudaDeviceSynchronize();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();
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

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      double_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      // double_kernel_x_rel_check<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.x, c.binsizex, c.binsizey, c.binsizez, c.type);
      // cudaDeviceSynchronize();
    }
    else {
      double_force_kernel_xdata<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, f, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }
    cudaDeviceSynchronize();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    return ev;
  }

  void test_kernel_launch(int ntotal) {
    printf("launch test kernel\n");

    //double* f_ptr = (double*)f.data();
    double* f_ptr;
    cudaMallocManaged(&f_ptr, (ntotal + 1) * sizeof(double));

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    printf("neighlist size : %d, pairstyle size : %d\n", sizeof(list), sizeof(c));

    //test_kernel<PairStyle><<<blocksPerGrid, threadsPerBlock>>>(ntotal, list, c, f, f_ptr);
    test_kernel<PairStyle, 0><<<blocksPerGrid, threadsPerBlock>>>(ntotal, f, f_ptr);
    cudaDeviceSynchronize();

    printf("f_ptr value : %lf\n", f_ptr[10]);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    compute_item_custom<0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &energy_virial) const {
    energy_virial += compute_item_custom<1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  /*KOKKOS_INLINE_FUNCTION
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
  }*/

};

template <class DeviceType, int EVFLAG>
__global__ void float_force_kernel_xdata(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const float* x_float_data = x_float.data();
    const float xtmp = x_float_data[3 * i + 0];
    const float ytmp = x_float_data[3 * i + 1];
    const float ztmp = x_float_data[3 * i + 2];
    //const float xtmp = x_float(i, 0);
    //const float ytmp = x_float(i, 1);
    //const float ztmp = x_float(i, 2);
    const X_FLOAT* x_data = x.data();
    //const X_FLOAT xtmp = x_data[3 * i + 0];
    //const X_FLOAT ytmp = x_data[3 * i + 1];
    //const X_FLOAT ztmp = x_data[3 * i + 2];

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    // float fxtmp = 0.0f;
    // float fytmp = 0.0f;
    // float fztmp = 0.0f;
    double fxtmp = 0.0;
    double fytmp = 0.0;
    double fztmp = 0.0;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = 1.0f;
      j &= NEIGHMASK;
      const float delx = xtmp - x_float_data[3 * j + 0];
      const float dely = ytmp - x_float_data[3 * j + 1];
      const float delz = ztmp - x_float_data[3 * j + 2];
      //const float delx = xtmp - x_data[3 * j + 0];
      //const float dely = ytmp - x_data[3 * j + 1];
      //const float delz = ztmp - x_data[3 * j + 2];
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq) {
        const float r2inv = 1.0/rsq;
        const float r6inv = r2inv*r2inv*r2inv;

        const float forcelj = r6inv *
          (48*r6inv - 24);

        const float fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv*(4.0 * (F_FLOAT)r6inv -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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

template <class DeviceType, int EVFLAG>
__global__ void float_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel, X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const float xtmp = x_float_rel(i, 0);
    const float ytmp = x_float_rel(i, 1);
    const float ztmp = x_float_rel(i, 2);
    const int itype = type(i);

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
      int j = ni & DIRNEIGHMASK;
      const float delx = xtmp - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely = ytmp - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz = ztmp - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq) {
        const float r2inv = 1.0/rsq;
        const float r6inv = r2inv*r2inv*r2inv;

        const float forcelj = r6inv *
          (48*r6inv - 24);

        const float fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv*(4.0 * (F_FLOAT)r6inv -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, NEIGH_SEP_STRATEGY NEIGH_STG = NO_NEIGH_SEP, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomFloat  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  //Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array;

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

  void contribute_custom() {
    //Kokkos::Experimental::contribute(c.f, dup_f);

    /*if (c.eflag_atom)
      Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

    if (c.vflag_atom)
      Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
    */
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n"); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // MemoryKokkos::realloc_kokkos
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
      if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
        printf("lazy init x_float\n"); fflush(stdout);
        fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float = fpair -> x_float;
        printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
        //printf("x extent : %d %d\n", (fpair -> x).extent(0), (fpair -> x).extent(1));
      }
    }

    //printf("float kernel launch part0\n"); fflush(stdout);

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    //printf("float kernel launch part1\n"); fflush(stdout);

    if (USE_RELATIVE_COORD) {
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float_rel = c.x_float_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_float_rel(i,0) = (float)(curr_x(i,0) - curr_bin_base(i,0));
        curr_x_float_rel(i,1) = (float)(curr_x(i,1) - curr_bin_base(i,1));
        curr_x_float_rel(i,2) = (float)(curr_x(i,2) - curr_bin_base(i,2));
      });
    }
    else {
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float = c.x_float;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_float(i,0) = (float)curr_x(i,0);
        curr_x_float(i,1) = (float)curr_x(i,1);
        curr_x_float(i,2) = (float)curr_x(i,2);
      });
    }

    Kokkos::fence();

    //printf("float kernel launch part2\n"); fflush(stdout);

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      float_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }
    else {
      float_force_kernel_xdata<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.x, c.type, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }

    cudaDeviceSynchronize();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    //printf("float kernel launch part3\n"); fflush(stdout);
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    //printf("launch cuda reduce kernel\n"); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
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
      if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)){
        printf("lazy init x_float\n");
        fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float = fpair -> x_float;
        printf("x_float extend : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
      }
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    //printf("float kernel launch reduce part1\n"); fflush(stdout);

    if (USE_RELATIVE_COORD) {
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float_rel = c.x_float_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_float_rel(i,0) = (float)(curr_x(i,0) - curr_bin_base(i,0));
        curr_x_float_rel(i,1) = (float)(curr_x(i,1) - curr_bin_base(i,1));
        curr_x_float_rel(i,2) = (float)(curr_x(i,2) - curr_bin_base(i,2));
      });
    }
    else {
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float = c.x_float;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_float(i,0) = (float)curr_x(i,0);
        curr_x_float(i,1) = (float)curr_x(i,1);
        curr_x_float(i,2) = (float)curr_x(i,2);
      });
    }

    Kokkos::fence();

    //printf("float kernel launch reduce part2\n"); fflush(stdout);

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      float_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, f, c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }
    else {
      float_force_kernel_xdata<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.x, c.type, f, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }

    cudaDeviceSynchronize();

    //printf("float kernel launch reduce part3\n"); fflush(stdout);

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    //fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    //printf("float kernel launch reduce part4\n"); fflush(stdout);

    return ev;
  }
};


template <class DeviceType>
__global__ void init_aos_xhalf_kernel(int ntotal, AoS_half* x_half, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type, 
  typename ArrayTypes<DeviceType>::t_x_array_randomread x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ntotal) {
    x_half[i].x[0] = __double2half(x(i,0));
    x_half[i].x[1] = __double2half(x(i,1));
    x_half[i].x[2] = __double2half(x(i,2));
    x_half[i].type = static_cast<short>(type(i));
  }
}

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

template <class DeviceType, int EVFLAG>
__global__ void half_force_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half, 
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const AoS_half half_data = x_half[i];
    const __half xtmp = half_data.x[0];
    const __half ytmp = half_data.x[1];
    const __half ztmp = half_data.x[2];

    const short itype = half_data.type;

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
      int j = neighbors_i(jj);
      const __half factor_lj = one_h;
      j &= NEIGHMASK;
      const AoS_half half_data_j = x_half[j];

      const __half delx = xtmp - half_data_j.x[0];
      const __half dely = ytmp - half_data_j.x[1];
      const __half delz = ztmp - half_data_j.x[2];

      const short jtype = half_data_j.type;
      const __half rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < __float2half(cutsq)) {
        const __half r2inv = one_h/rsq;
        const __half r6inv = r2inv*r2inv*r2inv;

        const __half forcelj = r6inv *
          (__float2half(48.0f)*r6inv - __float2half(24.0f));

        const __half fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv)*(4.0 * __half2float(r6inv) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx*delx) * __half2float(fpair);
            const E_FLOAT v1 = __half2float(dely*dely) * __half2float(fpair);
            const E_FLOAT v2 = __half2float(delz*delz) * __half2float(fpair);
            const E_FLOAT v3 = __half2float(delx*dely) * __half2float(fpair);
            const E_FLOAT v4 = __half2float(delx*delz) * __half2float(fpair);
            const E_FLOAT v5 = __half2float(dely*delz) * __half2float(fpair);

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

template <class DeviceType, int EVFLAG>
__global__ void half_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, half binsizex_h, half binsizey_h, half binsizez_h,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const AoS_half half_data = x_half_rel[i];
    const __half xtmp = half_data.x[0];
    const __half ytmp = half_data.x[1];
    const __half ztmp = half_data.x[2];

    const short itype = half_data.type;

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
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx = xtmp - half_data_j.x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex_h;
      const __half dely = ytmp - half_data_j.x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey_h;
      const __half delz = ztmp - half_data_j.x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez_h;

      const short jtype = half_data_j.type;
      const __half rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < __float2half(cutsq)) {
        const __half r2inv = one_h/rsq;
        const __half r6inv = r2inv*r2inv*r2inv;

        const __half forcelj = r6inv *
          (__float2half(48.0f)*r6inv - __float2half(24.0f));

        const __half fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv)*(4.0 * __half2float(r6inv) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx*delx) * __half2float(fpair);
            const E_FLOAT v1 = __half2float(dely*dely) * __half2float(fpair);
            const E_FLOAT v2 = __half2float(delz*delz) * __half2float(fpair);
            const E_FLOAT v3 = __half2float(delx*dely) * __half2float(fpair);
            const E_FLOAT v4 = __half2float(delx*delz) * __half2float(fpair);
            const E_FLOAT v5 = __half2float(dely*delz) * __half2float(fpair);

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

template <class DeviceType, int EVFLAG>
__global__ void half2_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, const half binsizex_h, const half binsizey_h, const half binsizez_h,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const AoS_half half_data = x_half_rel[i];
    const __half2 xtmp2 = __half2half2(half_data.x[0]);
    const __half2 ytmp2 = __half2half2(half_data.x[1]);
    const __half2 ztmp2 = __half2half2(half_data.x[2]);

    const short itype = half_data.type;

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    __half2 fxtmp2 = zero_h2;
    __half2 fytmp2 = zero_h2;
    __half2 fztmp2 = zero_h2;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    int jj;
    for (jj = 0; jj + 1 < jnum; jj += 2) {
      int ni1 = neighbors_i(jj);
      int ni2 = neighbors_i(jj + 1);
      const __half2 factor_lj2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half half_data_j1 = x_half_rel[j1];
      const AoS_half half_data_j2 = x_half_rel[j2];

      const __half2 delx2 = xtmp2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(binsizex_h);
      const __half2 dely2 = ytmp2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(binsizey_h);
      const __half2 delz2 = ztmp2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(binsizez_h);

      const short jtype1 = half_data_j1.type;
      const short jtype2 = half_data_j2.type;
      const __half2 rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;

      __half2 cmp_cut2 = __hle2(rsq2, __half2half2(__float2half(cutsq)));

      const __half2 r2inv2 = one_h2/rsq2;
      const __half2 r6inv2 = r2inv2*r2inv2*r2inv2;

      const __half2 forcelj2 = r6inv2 *
        (__half2half2(__float2half(48.0f))*r6inv2 - __half2half2(__float2half(24.0f)));

      const __half2 fpair2 = factor_lj2*forcelj2*r2inv2;

      fxtmp2 += delx2*fpair2*cmp_cut2;
      fytmp2 += dely2*fpair2*cmp_cut2;
      fztmp2 += delz2*fpair2*cmp_cut2;

      if (EVFLAG) {
        if (cmp_cut2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv2.x)*(4.0 * __half2float(r6inv2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx2.x*delx2.x) * __half2float(fpair2.x);
            const E_FLOAT v1 = __half2float(dely2.x*dely2.x) * __half2float(fpair2.x);
            const E_FLOAT v2 = __half2float(delz2.x*delz2.x) * __half2float(fpair2.x);
            const E_FLOAT v3 = __half2float(delx2.x*dely2.x) * __half2float(fpair2.x);
            const E_FLOAT v4 = __half2float(delx2.x*delz2.x) * __half2float(fpair2.x);
            const E_FLOAT v5 = __half2float(dely2.x*delz2.x) * __half2float(fpair2.x);

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
        if (cmp_cut2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv2.y)*(4.0 * __half2float(r6inv2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx2.y*delx2.y) * __half2float(fpair2.y);
            const E_FLOAT v1 = __half2float(dely2.y*dely2.y) * __half2float(fpair2.y);
            const E_FLOAT v2 = __half2float(delz2.y*delz2.y) * __half2float(fpair2.y);
            const E_FLOAT v3 = __half2float(delx2.y*dely2.y) * __half2float(fpair2.y);
            const E_FLOAT v4 = __half2float(delx2.y*delz2.y) * __half2float(fpair2.y);
            const E_FLOAT v5 = __half2float(dely2.y*delz2.y) * __half2float(fpair2.y);

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

    __half fxtmp = fxtmp2.x + fxtmp2.y;
    __half fytmp = fytmp2.x + fytmp2.y;
    __half fztmp = fztmp2.x + fztmp2.y;

    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    for (; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const __half factor_lj = one_h;
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx = half_data.x[0] - half_data_j.x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex_h;
      const __half dely = half_data.x[1] - half_data_j.x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey_h;
      const __half delz = half_data.x[2] - half_data_j.x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez_h;

      const short jtype = half_data_j.type;
      const __half rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < __float2half(cutsq)) {
        const __half r2inv = one_h/rsq;
        const __half r6inv = r2inv*r2inv*r2inv;

        const __half forcelj = r6inv *
          (__float2half(48.0f)*r6inv - __float2half(24.0f));

        const __half fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv)*(4.0 * __half2float(r6inv) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx*delx) * __half2float(fpair);
            const E_FLOAT v1 = __half2float(dely*dely) * __half2float(fpair);
            const E_FLOAT v2 = __half2float(delz*delz) * __half2float(fpair);
            const E_FLOAT v3 = __half2float(delx*dely) * __half2float(fpair);
            const E_FLOAT v4 = __half2float(delx*delz) * __half2float(fpair);
            const E_FLOAT v5 = __half2float(dely*delz) * __half2float(fpair);

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

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, NEIGH_SEP_STRATEGY NEIGH_STG = NO_NEIGH_SEP, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomHalf  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  //Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array;

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

  void contribute_custom() {
    //Kokkos::Experimental::contribute(c.f, dup_f);

    /*if (c.eflag_atom)
      Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

    if (c.vflag_atom)
      Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
    */
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n"); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // MemoryKokkos::realloc_kokkos
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
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half);
        }
        cudaMalloc((void**)&(fpair -> x_half), (fpair -> x).extent(0) * sizeof(AoS_half));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half = fpair -> x_half;
        printf("x_half extent : %d\n", fpair -> x_half_size);
      }
    }

    //printf("float kernel launch part0\n"); fflush(stdout);

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    //printf("float kernel launch part1\n"); fflush(stdout);
    
    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      init_aos_xhalf_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel, c.type, c.x, c.bin_base);
    }
    else {
      init_aos_xhalf_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half, c.type, c.x);
    }

    cudaDeviceSynchronize();

    threadsPerBlock = 128;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
#ifdef USE_HALF2
      half2_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
#else
      half_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
#endif
    }
    else {
      half_force_kernel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }

    cudaDeviceSynchronize();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    //printf("float kernel launch part3\n"); fflush(stdout);
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    //printf("launch cuda reduce kernel\n"); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
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
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half);
        }
        cudaMalloc((void**)&(fpair -> x_half), (fpair -> x).extent(0) * sizeof(AoS_half));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half = fpair -> x_half;
        printf("x_half extent : %d\n", fpair -> x_half_size);
      }
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
    else {
      init_aos_xhalf_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half, c.type, c.x);
    }

    cudaDeviceSynchronize();

    threadsPerBlock = 128;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
#ifdef USE_HALF2
      half2_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
#else
      half_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
#endif
    }
    else {
      half_force_kernel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }

    cudaDeviceSynchronize();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    return ev;
  }
};


template <class DeviceType>
__global__ void init_aos_xfhmix_kernel(int ntotal, AoS_half_xonly* x_half_xonly, 
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type, 
  typename ArrayTypes<DeviceType>::t_x_array_randomread x) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < ntotal) {
    double xtmp = x(i,0);
    double ytmp = x(i,1);
    double ztmp = x(i,2);
    x_float(i,0) = (float)xtmp;
    x_float(i,1) = (float)ytmp;
    x_float(i,2) = (float)ztmp;
    x_half_xonly[i].x[0] = __double2half(xtmp);
    x_half_xonly[i].x[1] = __double2half(ytmp);
    x_half_xonly[i].x[2] = __double2half(ztmp);
  }
}

template <class DeviceType>
__global__ void init_aos_xfhmix_rel_kernel(int ntotal, AoS_half_xonly* x_half_rel_xonly, 
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
    x_half_rel_xonly[i].x[0] = __double2half(xtmp);
    x_half_rel_xonly[i].x[1] = __double2half(ytmp);
    x_half_rel_xonly[i].x[2] = __double2half(ztmp);
  }
}

template <class DeviceType, int EVFLAG>
__global__ void fhmix_basic_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread fhcut_split,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int fh_cut = fhcut_split(i);
    const int jnum = d_numneigh(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    const __half xtmp_h = __float2half(xtmp_f);
    const __half ytmp_h = __float2half(ytmp_f);
    const __half ztmp_h = __float2half(ztmp_f);

    __half zero_h = __ushort_as_half((unsigned short)0x0000U);
    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    __half fxtmp_h = zero_h;
    __half fytmp_h = zero_h;
    __half fztmp_h = zero_h;

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      // const __half factor_lj_h = one_h;
      // const __half delx_h = xtmp_h - x_half_rel_xonly[j].x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * __float2half(binsizex);
      // const __half dely_h = ytmp_h - x_half_rel_xonly[j].x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * __float2half(binsizey);
      // const __half delz_h = ztmp_h - x_half_rel_xonly[j].x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * __float2half(binsizez);
      //const __half rsq_h = delx_h * delx_h + dely_h * dely_h + delz_h * delz_h;
      const __half factor_lj_h = one_h;
      // const __half delx_h = __float2half(__half2float(xtmp_h) - __half2float(x_half_rel_xonly[j].x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      // const __half dely_h = __float2half(__half2float(ytmp_h) - __half2float(x_half_rel_xonly[j].x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      // const __half delz_h = __float2half(__half2float(ztmp_h) - __half2float(x_half_rel_xonly[j].x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);
    
      const __half delx_h = xtmp_h - x_half_rel_xonly[j].x[0] - __float2half(((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      const __half dely_h = ytmp_h - x_half_rel_xonly[j].x[1] - __float2half(((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      const __half delz_h = ztmp_h - x_half_rel_xonly[j].x[2] - __float2half(((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);
    
      //const float rsq_hf = __half2float(delx_h) * __half2float(delx_h) + __half2float(dely_h) * __half2float(dely_h) + __half2float(delz_h) * __half2float(delz_h);
      //const float rsq_hf = delx_hf * delx_hf + dely_hf * dely_hf + delz_hf * delz_hf;
      const __half rsq_h = delx_h * delx_h + dely_h * dely_h + delz_h * delz_h;
      
      if (rsq_f < cutsq * 0.5f * 0.5f) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
      //else if(rsq_f < cutsq) {
      else if(rsq_h < __float2half(cutsq)) {
        // const __half r2inv_h = one_h / __float2half(rsq_hf);
        const __half r2inv_h = one_h / rsq_h;
        const __half r6inv_h = r2inv_h * r2inv_h * r2inv_h;

        const __half forcelj_h = r6inv_h *
          (__float2half(48.0f)*r6inv_h - __float2half(24.0f));

        const __half fpair_h = factor_lj_h*forcelj_h*r2inv_h;

        // fxtmp_f += __half2float(delx_h * fpair_h);
        // fytmp_f += __half2float(dely_h * fpair_h);
        // fztmp_f += __half2float(delz_h * fpair_h);
        fxtmp_h += delx_h * fpair_h;
        fytmp_h += dely_h * fpair_h;
        fztmp_h += delz_h * fpair_h;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h)*(4.0 * __half2float(r6inv_h) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h*delx_h) * __half2float(fpair_h);
            const E_FLOAT v1 = __half2float(dely_h*dely_h) * __half2float(fpair_h);
            const E_FLOAT v2 = __half2float(delz_h*delz_h) * __half2float(fpair_h);
            const E_FLOAT v3 = __half2float(delx_h*dely_h) * __half2float(fpair_h);
            const E_FLOAT v4 = __half2float(delx_h*delz_h) * __half2float(fpair_h);
            const E_FLOAT v5 = __half2float(dely_h*delz_h) * __half2float(fpair_h);

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
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    f(i,0) += __half2float(fxtmp_h);
    f(i,1) += __half2float(fytmp_h);
    f(i,2) += __half2float(fztmp_h);

    if (EVFLAG) {
      ev_array(ii) = ev;
    }
  }
}


template <class DeviceType, int EVFLAG>
__global__ void fhmix_half2_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread fhcut_split,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

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
    //for (jj = 0; jj < 0; jj++) {
    //for (jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    const __half2 xtmp_h2 = __half2half2(__float2half(xtmp_f));
    const __half2 ytmp_h2 = __half2half2(__float2half(ytmp_f));
    const __half2 ztmp_h2 = __half2half2(__float2half(ztmp_f));

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const int jnum = d_numneigh(i);

    __half2 fxtmp_h2 = zero_h2;
    __half2 fytmp_h2 = zero_h2;
    __half2 fztmp_h2 = zero_h2;

    // if((fh_cut ^ jnum) & 1) {
    //   printf("error : fh_cut ^ jnum & 1\n"); return;
    // }

    for (; jj + 1 < jnum; jj += 2) {
      int ni1 = neighbors_i(jj);
      int ni2 = neighbors_i(jj + 1);
      const __half2 factor_lj_h2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
      const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
      const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

      // const short jtype1 = half_data_j1.type;
      // const short jtype2 = half_data_j2.type;
      const __half2 rsq_h2 = delx_h2 * delx_h2 + dely_h2 * dely_h2 + delz_h2 * delz_h2;

      __half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));

      const __half2 r2inv_h2 = one_h2 / rsq_h2;
      const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;

      const __half2 forcelj_h2 = r6inv_h2 *
        (__half2half2(__float2half(48.0f))*r6inv_h2 - __half2half2(__float2half(24.0f)));

      const __half2 fpair_h2 = factor_lj_h2 * forcelj_h2 * r2inv_h2;

      fxtmp_h2 += delx_h2 * fpair_h2 * cmp_cut_h2;
      fytmp_h2 += dely_h2 * fpair_h2 * cmp_cut_h2;
      fztmp_h2 += delz_h2 * fpair_h2 * cmp_cut_h2;

      if (EVFLAG) {
        if (cmp_cut_h2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.x)*(4.0 * __half2float(r6inv_h2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.x * delx_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v1 = __half2float(dely_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v2 = __half2float(delz_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v3 = __half2float(delx_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v4 = __half2float(delx_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v5 = __half2float(dely_h2.x * delz_h2.x) * __half2float(fpair_h2.x);

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
        if (cmp_cut_h2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.y)*(4.0 * __half2float(r6inv_h2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.y * delx_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v1 = __half2float(dely_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v2 = __half2float(delz_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v3 = __half2float(delx_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v4 = __half2float(delx_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v5 = __half2float(dely_h2.y * delz_h2.y) * __half2float(fpair_h2.y);

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
    f(i,0) += __half2float(fxtmp_h2.x + fxtmp_h2.y);
    f(i,1) += __half2float(fytmp_h2.x + fytmp_h2.y);
    f(i,2) += __half2float(fztmp_h2.x + fztmp_h2.y);
  }
}

template <class DeviceType, int EVFLAG>
__global__ void fhmix_half2_force_kernel_x_rel_shared_mem(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread fhcut_split,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  /// init table f(x)=1.0/x 
  __shared__ __half inv_h[4096];
  for(ushort i = threadIdx.x; i < 4096; i += blockDim.x) {
    __half val = __ushort_as_half(0x4000 | i);
    if(val < __float2half(cutsq))
      inv_h[i] = __float2half(1.0f / __half2float(val));
    else 
      inv_h[i] = 0;
  }
  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

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
    //for (jj = 0; jj < 0; jj++) {
    //for (jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    const __half2 xtmp_h2 = __half2half2(__float2half(xtmp_f));
    const __half2 ytmp_h2 = __half2half2(__float2half(ytmp_f));
    const __half2 ztmp_h2 = __half2half2(__float2half(ztmp_f));

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const int jnum = d_numneigh(i);

    __half2 fxtmp_h2 = zero_h2;
    __half2 fytmp_h2 = zero_h2;
    __half2 fztmp_h2 = zero_h2;

    // if((fh_cut ^ jnum) & 1) {
    //   printf("error : fh_cut ^ jnum & 1\n"); return;
    // }

    for (; jj + 1 < jnum; jj += 2) {
      int ni1 = neighbors_i(jj);
      int ni2 = neighbors_i(jj + 1);
      const __half2 factor_lj_h2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
      const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
      const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

      // const short jtype1 = half_data_j1.type;
      // const short jtype2 = half_data_j2.type;
      const __half2 rsq_h2 = delx_h2 * delx_h2 + dely_h2 * dely_h2 + delz_h2 * delz_h2;

      //__half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));

      //const __half2 r2inv_h2 = one_h2 / rsq_h2;
      const __half2 r2inv_h2 = __halves2half2(inv_h[__half_as_ushort(rsq_h2.x) ^ 0x4000], inv_h[__half_as_ushort(rsq_h2.y) ^ 0x4000]);
      
      const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;

      const __half2 forcelj_h2 = r6inv_h2 *
        (__half2half2(__float2half(48.0f))*r6inv_h2 - __half2half2(__float2half(24.0f)));

      const __half2 fpair_h2 = factor_lj_h2 * forcelj_h2 * r2inv_h2;

      // fxtmp_h2 += delx_h2 * fpair_h2 * cmp_cut_h2;
      // fytmp_h2 += dely_h2 * fpair_h2 * cmp_cut_h2;
      // fztmp_h2 += delz_h2 * fpair_h2 * cmp_cut_h2;
      fxtmp_h2 += delx_h2 * fpair_h2;
      fytmp_h2 += dely_h2 * fpair_h2;
      fztmp_h2 += delz_h2 * fpair_h2;

      if (EVFLAG) {
        __half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));
        if (cmp_cut_h2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.x)*(4.0 * __half2float(r6inv_h2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.x * delx_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v1 = __half2float(dely_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v2 = __half2float(delz_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v3 = __half2float(delx_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v4 = __half2float(delx_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v5 = __half2float(dely_h2.x * delz_h2.x) * __half2float(fpair_h2.x);

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
        if (cmp_cut_h2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.y)*(4.0 * __half2float(r6inv_h2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.y * delx_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v1 = __half2float(dely_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v2 = __half2float(delz_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v3 = __half2float(delx_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v4 = __half2float(delx_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v5 = __half2float(dely_h2.y * delz_h2.y) * __half2float(fpair_h2.y);

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
    f(i,0) += __half2float(fxtmp_h2.x + fxtmp_h2.y);
    f(i,1) += __half2float(fytmp_h2.x + fytmp_h2.y);
    f(i,2) += __half2float(fztmp_h2.x + fztmp_h2.y);
  }
}

template <class DeviceType, int EVFLAG>
__global__ void fhmix_two_end_half2_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numfront, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half_xonly* x_half_rel_xonly, const half binsizex_h, const half binsizey_h, const half binsizez_h,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const half cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const AoS_half_xonly half_data = x_half_rel_xonly[i];
    const __half2 xtmp2 = __half2half2(half_data.x[0]);
    const __half2 ytmp2 = __half2half2(half_data.x[1]);
    const __half2 ztmp2 = __half2half2(half_data.x[2]);

    // const short itype = half_data.type;

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numfront(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numfront(i);

    __half2 fxtmp2 = zero_h2;
    __half2 fytmp2 = zero_h2;
    __half2 fztmp2 = zero_h2;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    int jj;
    for (jj = 0; jj < jnum; jj += 2) {
      int ni1 = neighbors_i(jj);
      int ni2 = neighbors_i(jj + 1);
      const __half2 factor_lj2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      const __half2 delx2 = xtmp2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(binsizex_h);
      const __half2 dely2 = ytmp2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(binsizey_h);
      const __half2 delz2 = ztmp2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(binsizez_h);

      // const short jtype1 = half_data_j1.type;
      // const short jtype2 = half_data_j2.type;
      const __half2 rsq2 = delx2*delx2 + dely2*dely2 + delz2*delz2;

      __half2 cmp_cut2 = __hle2(rsq2, __half2half2(cutsq));

      const __half2 r2inv2 = one_h2/rsq2;
      const __half2 r6inv2 = r2inv2*r2inv2*r2inv2;

      const __half2 forcelj2 = r6inv2 *
        (__half2half2(__float2half(48.0f))*r6inv2 - __half2half2(__float2half(24.0f)));

      const __half2 fpair2 = factor_lj2*forcelj2*r2inv2;

      fxtmp2 += delx2*fpair2*cmp_cut2;
      fytmp2 += dely2*fpair2*cmp_cut2;
      fztmp2 += delz2*fpair2*cmp_cut2;

      if (EVFLAG) {
        if (cmp_cut2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv2.x)*(4.0 * __half2float(r6inv2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx2.x*delx2.x) * __half2float(fpair2.x);
            const E_FLOAT v1 = __half2float(dely2.x*dely2.x) * __half2float(fpair2.x);
            const E_FLOAT v2 = __half2float(delz2.x*delz2.x) * __half2float(fpair2.x);
            const E_FLOAT v3 = __half2float(delx2.x*dely2.x) * __half2float(fpair2.x);
            const E_FLOAT v4 = __half2float(delx2.x*delz2.x) * __half2float(fpair2.x);
            const E_FLOAT v5 = __half2float(dely2.x*delz2.x) * __half2float(fpair2.x);

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
        if (cmp_cut2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv2.y)*(4.0 * __half2float(r6inv2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx2.y*delx2.y) * __half2float(fpair2.y);
            const E_FLOAT v1 = __half2float(dely2.y*dely2.y) * __half2float(fpair2.y);
            const E_FLOAT v2 = __half2float(delz2.y*delz2.y) * __half2float(fpair2.y);
            const E_FLOAT v3 = __half2float(delx2.y*dely2.y) * __half2float(fpair2.y);
            const E_FLOAT v4 = __half2float(delx2.y*delz2.y) * __half2float(fpair2.y);
            const E_FLOAT v5 = __half2float(dely2.y*delz2.y) * __half2float(fpair2.y);

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
    f(i,0) += __half2float(fxtmp2.x + fxtmp2.y);
    f(i,1) += __half2float(fytmp2.x + fytmp2.y);
    f(i,2) += __half2float(fztmp2.x + fztmp2.y);
  }
}

template <class DeviceType, int EVFLAG>
__global__ void fhmix_two_end_float_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numback, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, int maxneighs,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel, float binsizex, float binsizey, float binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const float xtmp = x_float_rel(i, 0);
    const float ytmp = x_float_rel(i, 1);
    const float ztmp = x_float_rel(i, 2);
    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), maxneighs, &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numback(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = jnum; jj < maxneighs; jj++) {
      int ni = neighbors_i(jj);
      const float factor_lj = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx = xtmp - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely = ytmp - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz = ztmp - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq) {
        const float r2inv = 1.0/rsq;
        const float r6inv = r2inv*r2inv*r2inv;

        const float forcelj = r6inv *
          (48*r6inv - 24);

        const float fpair = factor_lj*forcelj*r2inv;

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv*(4.0 * (F_FLOAT)r6inv -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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

template <class DeviceType, int EVFLAG>
__global__ void fhmix_two_end_half2_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numfront, typename ArrayTypes<DeviceType>::t_int_1d d_numback, 
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, int maxneighs,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), maxneighs, &d_neighbors(i,1)-&d_neighbors(i,0));
    //const int fh_cut = fhcut_split(i);
    //const int jnum = d_numneigh(i);
    const int num_back = d_numback(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = num_back; jj < maxneighs; jj++) {
    //for (jj = 0; jj < 0; jj++) {
    //for (jj = 0; jj < jnum; jj++) {
      // int ni = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    const __half2 xtmp_h2 = __half2half2(__float2half(xtmp_f));
    const __half2 ytmp_h2 = __half2half2(__float2half(ytmp_f));
    const __half2 ztmp_h2 = __half2half2(__float2half(ztmp_f));

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    //const int jnum = d_numneigh(i);
    const int num_front = d_numfront(i);
    __half2 fxtmp_h2 = zero_h2;
    __half2 fytmp_h2 = zero_h2;
    __half2 fztmp_h2 = zero_h2;

    // if((fh_cut ^ jnum) & 1) {
    //   printf("error : fh_cut ^ jnum & 1\n"); return;
    // }

    for (int jj = 0; jj < num_front; jj += 2) {
      // int ni1 = neighbors_i(jj);
      // int ni2 = neighbors_i(jj + 1);
      const int *neighbors_i_raw_ptr_jj = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      const int *neighbors_i_raw_ptr_jj_1 = &neighbors_i._firstneigh[(size_t) (jj+1) * neighbors_i._stride];
      int ni1 = __ldcs(neighbors_i_raw_ptr_jj);
      int ni2 = __ldcs(neighbors_i_raw_ptr_jj_1);

      const __half2 factor_lj_h2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
      const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
      const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

      // const short jtype1 = half_data_j1.type;
      // const short jtype2 = half_data_j2.type;
      const __half2 rsq_h2 = delx_h2 * delx_h2 + dely_h2 * dely_h2 + delz_h2 * delz_h2;

      __half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));

      const __half2 r2inv_h2 = one_h2 / rsq_h2;
      const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;

      const __half2 forcelj_h2 = r6inv_h2 *
        (__half2half2(__float2half(48.0f))*r6inv_h2 - __half2half2(__float2half(24.0f)));

      const __half2 fpair_h2 = factor_lj_h2 * forcelj_h2 * r2inv_h2;

      fxtmp_h2 += delx_h2 * fpair_h2 * cmp_cut_h2;
      fytmp_h2 += dely_h2 * fpair_h2 * cmp_cut_h2;
      fztmp_h2 += delz_h2 * fpair_h2 * cmp_cut_h2;

      if (EVFLAG) {
        if (cmp_cut_h2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.x)*(4.0 * __half2float(r6inv_h2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.x * delx_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v1 = __half2float(dely_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v2 = __half2float(delz_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v3 = __half2float(delx_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v4 = __half2float(delx_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v5 = __half2float(dely_h2.x * delz_h2.x) * __half2float(fpair_h2.x);

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
        if (cmp_cut_h2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.y)*(4.0 * __half2float(r6inv_h2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.y * delx_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v1 = __half2float(dely_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v2 = __half2float(delz_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v3 = __half2float(delx_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v4 = __half2float(delx_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v5 = __half2float(dely_h2.y * delz_h2.y) * __half2float(fpair_h2.y);

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
    f(i,0) += __half2float(fxtmp_h2.x + fxtmp_h2.y);
    f(i,1) += __half2float(fytmp_h2.x + fytmp_h2.y);
    f(i,2) += __half2float(fztmp_h2.x + fztmp_h2.y);
  }
}

#define BLOCK_SIZE 512

template <class DeviceType, int EVFLAG>
__global__ void __launch_bounds__(BLOCK_SIZE, 4) fhmix_int2_half2_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_int2, 
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, typename ArrayTypes<DeviceType>::t_neighbors_2d_int2 d_neighbors_int2,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    //const int fh_cut = fhcut_split(i);
    const int jnum = d_numneigh(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      // int ni = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    const __half2 xtmp_h2 = __half2half2(__float2half(xtmp_f));
    const __half2 ytmp_h2 = __half2half2(__float2half(ytmp_f));
    const __half2 ztmp_h2 = __half2half2(__float2half(ztmp_f));

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const AtomNeighborsConst_int2 neighbors_i_int2 = 
      AtomNeighborsConst_int2(&d_neighbors_int2(i,0), d_numneigh_int2(i), &d_neighbors_int2(i,1)-&d_neighbors_int2(i,0));
    const int jnum_int2 = d_numneigh_int2(i) >> 1;
    // const int num_front = d_numfront(i);
    __half2 fxtmp_h2 = zero_h2;
    __half2 fytmp_h2 = zero_h2;
    __half2 fztmp_h2 = zero_h2;

    // if((fh_cut ^ jnum) & 1) {
    //   printf("error : fh_cut ^ jnum & 1\n"); return;
    // }

    for (int jj = 0; jj < jnum_int2; jj ++) {
      // int ni1 = neighbors_i(jj);
      // int ni2 = neighbors_i(jj + 1);
      const int2 *neighbors_i_raw_ptr_jj = &neighbors_i_int2._firstneigh[(size_t) jj * neighbors_i_int2._stride];
      int2 ni = __ldcs(neighbors_i_raw_ptr_jj);
      int ni1 = ni.x;
      int ni2 = ni.y;
      // const int *neighbors_i_raw_ptr_jj = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      // const int *neighbors_i_raw_ptr_jj_1 = &neighbors_i._firstneigh[(size_t) (jj+1) * neighbors_i._stride];
      // int ni1 = __ldcs(neighbors_i_raw_ptr_jj);
      // int ni2 = __ldcs(neighbors_i_raw_ptr_jj_1);

      const __half2 factor_lj_h2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      //__half2 delx_h2, dely_h2, delz_h2;
      // delx_h2.x = __float2half(xtmp_f - __half2float(half_data_j1.x[0]) - ((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      // delx_h2.y = __float2half(xtmp_f - __half2float(half_data_j2.x[0]) - ((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      // dely_h2.x = __float2half(ytmp_f - __half2float(half_data_j1.x[1]) - ((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      // dely_h2.y = __float2half(ytmp_f - __half2float(half_data_j2.x[1]) - ((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      // delz_h2.x = __float2half(ztmp_f - __half2float(half_data_j1.x[2]) - ((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);
      // delz_h2.y = __float2half(ztmp_f - __half2float(half_data_j2.x[2]) - ((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);

      const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
      const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
      const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

      // const short jtype1 = half_data_j1.type;
      // const short jtype2 = half_data_j2.type;
      const __half2 rsq_h2 = delx_h2 * delx_h2 + dely_h2 * dely_h2 + delz_h2 * delz_h2;

      __half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));

      const __half2 r2inv_h2 = one_h2 / rsq_h2;
      const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;

      const __half2 forcelj_h2 = r6inv_h2 *
        (__half2half2(__float2half(48.0f))*r6inv_h2 - __half2half2(__float2half(24.0f)));

      const __half2 fpair_h2 = factor_lj_h2 * forcelj_h2 * r2inv_h2;

      fxtmp_h2 += delx_h2 * fpair_h2 * cmp_cut_h2;
      fytmp_h2 += dely_h2 * fpair_h2 * cmp_cut_h2;
      fztmp_h2 += delz_h2 * fpair_h2 * cmp_cut_h2;

      if (EVFLAG) {
        if (cmp_cut_h2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.x)*(4.0 * __half2float(r6inv_h2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.x * delx_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v1 = __half2float(dely_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v2 = __half2float(delz_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v3 = __half2float(delx_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v4 = __half2float(delx_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v5 = __half2float(dely_h2.x * delz_h2.x) * __half2float(fpair_h2.x);

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
        if (cmp_cut_h2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.y)*(4.0 * __half2float(r6inv_h2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.y * delx_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v1 = __half2float(dely_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v2 = __half2float(delz_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v3 = __half2float(delx_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v4 = __half2float(delx_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v5 = __half2float(dely_h2.y * delz_h2.y) * __half2float(fpair_h2.y);

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
    f(i,0) += __half2float(fxtmp_h2.x + fxtmp_h2.y);
    f(i,1) += __half2float(fytmp_h2.x + fytmp_h2.y);
    f(i,2) += __half2float(fztmp_h2.x + fztmp_h2.y);
  }
}

/*template <class DeviceType, int EVFLAG>
__global__ void fhmix_int2_half2_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_int2, 
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, typename ArrayTypes<DeviceType>::t_neighbors_2d_int2 d_neighbors_int2,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    //const int fh_cut = fhcut_split(i);
    const int jnum = d_numneigh(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      // int ni = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    // const __half2 xtmp_h2 = __half2half2(__float2half(xtmp_f));
    // const __half2 ytmp_h2 = __half2half2(__float2half(ytmp_f));
    // const __half2 ztmp_h2 = __half2half2(__float2half(ztmp_f));

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const AtomNeighborsConst_int2 neighbors_i_int2 = 
      AtomNeighborsConst_int2(&d_neighbors_int2(i,0), d_numneigh_int2(i), &d_neighbors_int2(i,1)-&d_neighbors_int2(i,0));
    const int jnum_int2 = d_numneigh_int2(i) >> 1;
    // const int num_front = d_numfront(i);
    __half2 fxtmp_h2 = zero_h2;
    __half2 fytmp_h2 = zero_h2;
    __half2 fztmp_h2 = zero_h2;
    float2 fxtmp_f2 = make_float2(0.0, 0.0);
    float2 fytmp_f2 = make_float2(0.0, 0.0);
    float2 fztmp_f2 = make_float2(0.0, 0.0);

    // if((fh_cut ^ jnum) & 1) {
    //   printf("error : fh_cut ^ jnum & 1\n"); return;
    // }

    for (int jj = 0; jj < jnum_int2; jj ++) {
      // int ni1 = neighbors_i(jj);
      // int ni2 = neighbors_i(jj + 1);
      const int2 *neighbors_i_raw_ptr_jj = &neighbors_i_int2._firstneigh[(size_t) jj * neighbors_i_int2._stride];
      int2 ni = __ldcs(neighbors_i_raw_ptr_jj);
      int ni1 = ni.x;
      int ni2 = ni.y;

      const __half2 factor_lj_h2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      __half2 delx_h2, dely_h2, delz_h2;
      delx_h2.x = __float2half(xtmp_f - __half2float(half_data_j1.x[0]) - ((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      delx_h2.y = __float2half(xtmp_f - __half2float(half_data_j2.x[0]) - ((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      dely_h2.x = __float2half(ytmp_f - __half2float(half_data_j1.x[1]) - ((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      dely_h2.y = __float2half(ytmp_f - __half2float(half_data_j2.x[1]) - ((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      delz_h2.x = __float2half(ztmp_f - __half2float(half_data_j1.x[2]) - ((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);
      delz_h2.y = __float2half(ztmp_f - __half2float(half_data_j2.x[2]) - ((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);

      float2 delx_f2, dely_f2, delz_f2;
      delx_f2.x = xtmp_f - __half2float(half_data_j1.x[0]) - ((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      delx_f2.y = xtmp_f - __half2float(half_data_j2.x[0]) - ((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      dely_f2.x = ytmp_f - __half2float(half_data_j1.x[1]) - ((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      dely_f2.y = ytmp_f - __half2float(half_data_j2.x[1]) - ((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      delz_f2.x = ztmp_f - __half2float(half_data_j1.x[2]) - ((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      delz_f2.y = ztmp_f - __half2float(half_data_j2.x[2]) - ((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;

      //const __half2 rsq_h2 = delx_h2 * delx_h2 + dely_h2 * dely_h2 + delz_h2 * delz_h2;
      float2 rsq_f2;
      rsq_f2.x = delx_f2.x * delx_f2.x + dely_f2.x * dely_f2.x + delz_f2.x * delz_f2.x;
      rsq_f2.y = delx_f2.y * delx_f2.y + dely_f2.y * dely_f2.y + delz_f2.y * delz_f2.y;
      const __half2 rsq_h2 = __halves2half2(__float2half(rsq_f2.x), __float2half(rsq_f2.y));

      __half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));
      //float2 cmp_cut_f2 = make_float2(rsq_f2.x < cutsq, rsq_f2.y < cutsq); 

      const __half2 r2inv_h2 = one_h2 / rsq_h2;
      const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;
      const float2 r2inv_f2 = make_float2(1.0 / rsq_f2.x, 1.0 / rsq_f2.y);
      const float2 r6inv_f2 = make_float2(r2inv_f2.x * r2inv_f2.x * r2inv_f2.x, r2inv_f2.y * r2inv_f2.y * r2inv_f2.y);

      const __half2 forcelj_h2 = r6inv_h2 *
        (__half2half2(__float2half(48.0f))*r6inv_h2 - __half2half2(__float2half(24.0f)));
      const float2 forcelj_f2 = make_float2(r6inv_f2.x * (48.0f * r6inv_f2.x - 24.0f), r6inv_f2.y * (48.0f * r6inv_f2.y - 24.0f));

      const __half2 fpair_h2 = factor_lj_h2 * forcelj_h2 * r2inv_h2;
      const float2 fpair_f2 = make_float2(1.0f * forcelj_f2.x * r2inv_f2.x, 1.0f * forcelj_f2.y * r2inv_f2.y);

      fxtmp_h2 += delx_h2 * fpair_h2 * cmp_cut_h2;
      fytmp_h2 += dely_h2 * fpair_h2 * cmp_cut_h2;
      fztmp_h2 += delz_h2 * fpair_h2 * cmp_cut_h2;

      // fxtmp_f2.x += delx_f2.x * fpair_f2.x * cmp_cut_f2.x;
      // fxtmp_f2.y += delx_f2.y * fpair_f2.y * cmp_cut_f2.y;
      // fytmp_f2.x += dely_f2.x * fpair_f2.x * cmp_cut_f2.x;
      // fytmp_f2.y += dely_f2.y * fpair_f2.y * cmp_cut_f2.y;
      // fztmp_f2.x += delz_f2.x * fpair_f2.x * cmp_cut_f2.x;
      // fztmp_f2.y += delz_f2.y * fpair_f2.y * cmp_cut_f2.y;

      // fxtmp_f2.x += __half2float(delx_h2.x * fpair_h2.x) * cmp_cut_f2.x;
      // fxtmp_f2.y += __half2float(delx_h2.y * fpair_h2.y) * cmp_cut_f2.y;
      // fytmp_f2.x += __half2float(dely_h2.x * fpair_h2.x) * cmp_cut_f2.x;
      // fytmp_f2.y += __half2float(dely_h2.y * fpair_h2.y) * cmp_cut_f2.y;
      // fztmp_f2.x += __half2float(delz_h2.x * fpair_h2.x) * cmp_cut_f2.x;
      // fztmp_f2.y += __half2float(delz_h2.y * fpair_h2.y) * cmp_cut_f2.y;

      // fxtmp_h2.x += delx_h2.x * __float2half(fpair_f2.x) * cmp_cut_h2.x;
      // fxtmp_h2.y += delx_h2.y * __float2half(fpair_f2.y) * cmp_cut_h2.y;
      // fytmp_h2.x += dely_h2.x * __float2half(fpair_f2.x) * cmp_cut_h2.x;
      // fytmp_h2.y += dely_h2.y * __float2half(fpair_f2.y) * cmp_cut_h2.y;
      // fztmp_h2.x += delz_h2.x * __float2half(fpair_f2.x) * cmp_cut_h2.x;
      // fztmp_h2.y += delz_h2.y * __float2half(fpair_f2.y) * cmp_cut_h2.y;


      if (EVFLAG) {
        if (cmp_cut_h2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.x)*(4.0 * __half2float(r6inv_h2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.x * delx_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v1 = __half2float(dely_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v2 = __half2float(delz_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v3 = __half2float(delx_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v4 = __half2float(delx_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v5 = __half2float(dely_h2.x * delz_h2.x) * __half2float(fpair_h2.x);

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
        if (cmp_cut_h2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.y)*(4.0 * __half2float(r6inv_h2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.y * delx_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v1 = __half2float(dely_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v2 = __half2float(delz_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v3 = __half2float(delx_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v4 = __half2float(delx_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v5 = __half2float(dely_h2.y * delz_h2.y) * __half2float(fpair_h2.y);

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
    f(i,0) += __half2float(fxtmp_h2.x + fxtmp_h2.y);
    f(i,1) += __half2float(fytmp_h2.x + fytmp_h2.y);
    f(i,2) += __half2float(fztmp_h2.x + fztmp_h2.y);
    // f(i, 0) += fxtmp_f2.x + fxtmp_f2.y;
    // f(i, 1) += fytmp_f2.x + fytmp_f2.y;
    // f(i, 2) += fztmp_f2.x + fztmp_f2.y;
  }
}
*/

template <class DeviceType, int EVFLAG>
__global__ void fhmix_int2_half2_force_kernel_x_rel_shared_mem(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_int2, 
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, typename ArrayTypes<DeviceType>::t_neighbors_2d_int2 d_neighbors_int2,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  /// init table f(x)=(1.0/x)^3 * (48*(1.0/x)^3 - 24) * (1.0/x)
  __shared__ __half fpair_h[3600];
  for(ushort i = threadIdx.x; i < 3600; i += blockDim.x) {
    __half val = __ushort_as_half(0x4000 | i);
    float inv_f = 1.0f / __half2float(val);
    float inv3_f = inv_f * inv_f * inv_f;
    if(val < __float2half(cutsq))
      fpair_h[i] = __float2half(inv3_f * (48.0f * inv3_f - 24.0f) * inv_f);
    else 
      fpair_h[i] = 0;
  }
  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    //const int fh_cut = fhcut_split(i);
    const int jnum = d_numneigh(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      // int ni = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
    // f(i,0) += (F_FLOAT)fxtmp_f;
    // f(i,1) += (F_FLOAT)fytmp_f;
    // f(i,2) += (F_FLOAT)fztmp_f;

    // const __half2 xtmp_h2 = __half2half2(__float2half(xtmp_f));
    // const __half2 ytmp_h2 = __half2half2(__float2half(ytmp_f));
    // const __half2 ztmp_h2 = __half2half2(__float2half(ztmp_f));

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const AtomNeighborsConst_int2 neighbors_i_int2 = 
      AtomNeighborsConst_int2(&d_neighbors_int2(i,0), d_numneigh_int2(i), &d_neighbors_int2(i,1)-&d_neighbors_int2(i,0));
    const int jnum_int2 = d_numneigh_int2(i) >> 1;
    // const int num_front = d_numfront(i);
    __half2 fxtmp_h2 = zero_h2;
    __half2 fytmp_h2 = zero_h2;
    __half2 fztmp_h2 = zero_h2;

    // if((fh_cut ^ jnum) & 1) {
    //   printf("error : fh_cut ^ jnum & 1\n"); return;
    // }

    for (int jj = 0; jj < jnum_int2; jj ++) {
      // int ni1 = neighbors_i(jj);
      // int ni2 = neighbors_i(jj + 1);
      const int2 *neighbors_i_raw_ptr_jj = &neighbors_i_int2._firstneigh[(size_t) jj * neighbors_i_int2._stride];
      int2 ni = __ldcs(neighbors_i_raw_ptr_jj);
      int ni1 = ni.x;
      int ni2 = ni.y;
      // const int *neighbors_i_raw_ptr_jj = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      // const int *neighbors_i_raw_ptr_jj_1 = &neighbors_i._firstneigh[(size_t) (jj+1) * neighbors_i._stride];
      // int ni1 = __ldcs(neighbors_i_raw_ptr_jj);
      // int ni2 = __ldcs(neighbors_i_raw_ptr_jj_1);

      const __half2 factor_lj_h2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      __half2 delx_h2, dely_h2, delz_h2;
      delx_h2.x = __float2half(xtmp_f - __half2float(half_data_j1.x[0]) - ((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      delx_h2.y = __float2half(xtmp_f - __half2float(half_data_j2.x[0]) - ((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      dely_h2.x = __float2half(ytmp_f - __half2float(half_data_j1.x[1]) - ((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      dely_h2.y = __float2half(ytmp_f - __half2float(half_data_j2.x[1]) - ((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      delz_h2.x = __float2half(ztmp_f - __half2float(half_data_j1.x[2]) - ((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);
      delz_h2.y = __float2half(ztmp_f - __half2float(half_data_j2.x[2]) - ((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);

      // const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
      //   - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
      //                    __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
      // const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
      //   - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
      //                    __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
      // const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
      //   - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
      //                    __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

      // const short jtype1 = half_data_j1.type;
      // const short jtype2 = half_data_j2.type;
      const __half2 rsq_h2 = delx_h2 * delx_h2 + dely_h2 * dely_h2 + delz_h2 * delz_h2;

      //__half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));

      //const __half2 r2inv_h2 = one_h2 / rsq_h2;
      //const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;

      //const __half2 forcelj_h2 = r6inv_h2 *
      //  (__half2half2(__float2half(48.0f))*r6inv_h2 - __half2half2(__float2half(24.0f)));

      //const __half2 fpair_h2 = factor_lj_h2 * forcelj_h2 * r2inv_h2;
      const __half2 fpair_h2 = factor_lj_h2 * __halves2half2(fpair_h[__half_as_ushort(rsq_h2.x) ^ 0x4000], fpair_h[__half_as_ushort(rsq_h2.y) ^ 0x4000]);

      // fxtmp_h2 += delx_h2 * fpair_h2 * cmp_cut_h2;
      // fytmp_h2 += dely_h2 * fpair_h2 * cmp_cut_h2;
      // fztmp_h2 += delz_h2 * fpair_h2 * cmp_cut_h2;

      // fxtmp_h2 += delx_h2 * fpair_h2;
      // fytmp_h2 += dely_h2 * fpair_h2;
      // fztmp_h2 += delz_h2 * fpair_h2;
      fxtmp_f += __half2float(delx_h2.x * fpair_h2.x + delx_h2.y * fpair_h2.y);
      fytmp_f += __half2float(dely_h2.x * fpair_h2.x + dely_h2.y * fpair_h2.y);
      fztmp_f += __half2float(delz_h2.x * fpair_h2.x + delz_h2.y * fpair_h2.y);

      if (EVFLAG) {
        __half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));
        const __half2 r2inv_h2 = one_h2 / rsq_h2;
        const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;

        if (cmp_cut_h2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.x)*(4.0 * __half2float(r6inv_h2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.x * delx_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v1 = __half2float(dely_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v2 = __half2float(delz_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v3 = __half2float(delx_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v4 = __half2float(delx_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v5 = __half2float(dely_h2.x * delz_h2.x) * __half2float(fpair_h2.x);

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
        if (cmp_cut_h2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.y)*(4.0 * __half2float(r6inv_h2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.y * delx_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v1 = __half2float(dely_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v2 = __half2float(delz_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v3 = __half2float(delx_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v4 = __half2float(delx_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v5 = __half2float(dely_h2.y * delz_h2.y) * __half2float(fpair_h2.y);

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
    // f(i,0) += __half2float(fxtmp_h2.x + fxtmp_h2.y);
    // f(i,1) += __half2float(fytmp_h2.x + fytmp_h2.y);
    // f(i,2) += __half2float(fztmp_h2.x + fztmp_h2.y);
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;
  }
}


template <class DeviceType, int EVFLAG>
__global__ void __launch_bounds__(BLOCK_SIZE, 4) fhmix_expr_optimized(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_int2, 
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, typename ArrayTypes<DeviceType>::t_neighbors_2d_int2 d_neighbors_int2,
  AoS_half_xonly* x_half_rel_xonly, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  const float binsizex, const float binsizey, const float binsizez,
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_f_array f, Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, const float cutsq) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;
    const int i = d_ilist(ii);
    
    const float xtmp_f = x_float_rel(i, 0);
    const float ytmp_f = x_float_rel(i, 1);
    const float ztmp_f = x_float_rel(i, 2);

    const int itype = type(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    //const int fh_cut = fhcut_split(i);
    const int jnum = d_numneigh(i);

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      // int ni = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj_f = 1.0f;
      int j = ni & DIRNEIGHMASK;
      const float delx_f = xtmp_f - x_float_rel(j, 0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const float dely_f = ytmp_f - x_float_rel(j, 1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const float delz_f = ztmp_f - x_float_rel(j, 2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq_f < cutsq) {
        const float r2inv_f = 1.0f / rsq_f;
        const float r6inv_f = r2inv_f * r2inv_f * r2inv_f;

        const float forcelj_f = r6inv_f *
          (48*r6inv_f - 24);

        const float fpair_f = factor_lj_f * forcelj_f * r2inv_f;

        fxtmp_f += delx_f * fpair_f;
        fytmp_f += dely_f * fpair_f;
        fztmp_f += delz_f * fpair_f;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = (F_FLOAT)r6inv_f * (4.0 * (F_FLOAT)r6inv_f -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
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
    f(i,0) += (F_FLOAT)fxtmp_f;
    f(i,1) += (F_FLOAT)fytmp_f;
    f(i,2) += (F_FLOAT)fztmp_f;

    const __half2 xtmp_h2 = __half2half2(__float2half(xtmp_f));
    const __half2 ytmp_h2 = __half2half2(__float2half(ytmp_f));
    const __half2 ztmp_h2 = __half2half2(__float2half(ztmp_f));

    __half2 zero_h2 = __half2half2(__ushort_as_half((unsigned short)0x0000U));
    __half2 one_h2 = __half2half2(__ushort_as_half((unsigned short)0x3C00U));

    const AtomNeighborsConst_int2 neighbors_i_int2 = 
      AtomNeighborsConst_int2(&d_neighbors_int2(i,0), d_numneigh_int2(i), &d_neighbors_int2(i,1)-&d_neighbors_int2(i,0));
    const int jnum_int2 = d_numneigh_int2(i) >> 1;
    // const int num_front = d_numfront(i);
    __half2 fxtmp_h2 = zero_h2;
    __half2 fytmp_h2 = zero_h2;
    __half2 fztmp_h2 = zero_h2;

    // if((fh_cut ^ jnum) & 1) {
    //   printf("error : fh_cut ^ jnum & 1\n"); return;
    // }

    for (int jj = 0; jj < jnum_int2; jj ++) {
      // int ni1 = neighbors_i(jj);
      // int ni2 = neighbors_i(jj + 1);
      const int2 *neighbors_i_raw_ptr_jj = &neighbors_i_int2._firstneigh[(size_t) jj * neighbors_i_int2._stride];
      int2 ni = __ldcs(neighbors_i_raw_ptr_jj);
      int ni1 = ni.x;
      int ni2 = ni.y;
      // const int *neighbors_i_raw_ptr_jj = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      // const int *neighbors_i_raw_ptr_jj_1 = &neighbors_i._firstneigh[(size_t) (jj+1) * neighbors_i._stride];
      // int ni1 = __ldcs(neighbors_i_raw_ptr_jj);
      // int ni2 = __ldcs(neighbors_i_raw_ptr_jj_1);

      const __half2 factor_lj_h2 = one_h2;
      int j1 = ni1 & DIRNEIGHMASK;
      int j2 = ni2 & DIRNEIGHMASK;
      const AoS_half_xonly half_data_j1 = x_half_rel_xonly[j1];
      const AoS_half_xonly half_data_j2 = x_half_rel_xonly[j2];

      const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
      const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
      const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
        - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
                         __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

      // const short jtype1 = half_data_j1.type;
      // const short jtype2 = half_data_j2.type;
      const __half2 rsq_h2 = delx_h2 * delx_h2 + dely_h2 * dely_h2 + delz_h2 * delz_h2;

      __half2 cmp_cut_h2 = __hle2(rsq_h2, __half2half2(__float2half(cutsq)));

      const __half2 r2inv_h2 = one_h2 / rsq_h2;
      const __half2 r6inv_h2 = r2inv_h2 * r2inv_h2 * r2inv_h2;

      const __half2 forcelj_h2 = r6inv_h2 *
        (__half2half2(__float2half(48.0f))*r6inv_h2 - __half2half2(__float2half(24.0f)));

      const __half2 fpair_h2 = factor_lj_h2 * forcelj_h2 * r2inv_h2;

      fxtmp_h2 += delx_h2 * fpair_h2 * cmp_cut_h2;
      fytmp_h2 += dely_h2 * fpair_h2 * cmp_cut_h2;
      fztmp_h2 += delz_h2 * fpair_h2 * cmp_cut_h2;

      if (EVFLAG) {
        if (cmp_cut_h2.x) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.x)*(4.0 * __half2float(r6inv_h2.x) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.x * delx_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v1 = __half2float(dely_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v2 = __half2float(delz_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v3 = __half2float(delx_h2.x * dely_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v4 = __half2float(delx_h2.x * delz_h2.x) * __half2float(fpair_h2.x);
            const E_FLOAT v5 = __half2float(dely_h2.x * delz_h2.x) * __half2float(fpair_h2.x);

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
        if (cmp_cut_h2.y) {
          F_FLOAT evdwl = 0.0;
          if (eflag) {
            evdwl = __half2float(r6inv_h2.y)*(4.0 * __half2float(r6inv_h2.y) -
                          4.0) -
                          0.0;
            ev.evdwl += 0.5*evdwl;
          }

          //if (vflag_either) ev_tally_double(ev,fpair,delx,dely,delz,vflag_either,vflag_global);
          if (vflag_either) {
            const E_FLOAT v0 = __half2float(delx_h2.y * delx_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v1 = __half2float(dely_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v2 = __half2float(delz_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v3 = __half2float(delx_h2.y * dely_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v4 = __half2float(delx_h2.y * delz_h2.y) * __half2float(fpair_h2.y);
            const E_FLOAT v5 = __half2float(dely_h2.y * delz_h2.y) * __half2float(fpair_h2.y);

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
    f(i,0) += __half2float(fxtmp_h2.x + fxtmp_h2.y);
    f(i,1) += __half2float(fytmp_h2.x + fytmp_h2.y);
    f(i,2) += __half2float(fztmp_h2.x + fztmp_h2.y);
  }
}


// float and half mix precision kernel
template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, NEIGH_SEP_STRATEGY NEIGH_STG = NO_NEIGH_SEP, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomFhmix  {
  typedef typename PairStyle::device_type device_type ;
  typedef ArrayTypes<device_type> AT;

  // Reduction type, contains evdwl, ecoul and virial[6]
  typedef EV_FLOAT value_type;

  //Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array;

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

  void contribute_custom() {
    //Kokkos::Experimental::contribute(c.f, dup_f);

    /*if (c.eflag_atom)
      Kokkos::Experimental::contribute(c.d_eatom, dup_eatom);

    if (c.vflag_atom)
      Kokkos::Experimental::contribute(c.d_vatom, dup_vatom);
    */
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    // printf("launch cuda kernel, ntotal : %d, f.extent(0) : %d\n", ntotal, f.extent(0)); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // MemoryKokkos::realloc_kokkos
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
        printf("lazy init x_half_rel_xonly\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_rel_xonly);
        }
        cudaMalloc((void**)&(fpair -> x_half_rel_xonly), (fpair -> x).extent(0) * sizeof(AoS_half_xonly));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_rel_xonly = fpair -> x_half_rel_xonly;
        printf("x_half_rel_xonly extent : %d\n", fpair -> x_half_size);
      }
    }
    else {
      if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
        printf("lazy init x_float\n"); fflush(stdout);
        fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float = fpair -> x_float;
        printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
      }
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half_xonly\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_xonly);
        }
        cudaMalloc((void**)&(fpair -> x_half_xonly), (fpair -> x).extent(0) * sizeof(AoS_half_xonly));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_xonly = fpair -> x_half_xonly;
        printf("x_half_xonly extent : %d\n", fpair -> x_half_size);
      }
    }

    //printf("float kernel launch part0\n"); fflush(stdout);

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    //printf("float kernel launch part1\n"); fflush(stdout);
    
    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      init_aos_xfhmix_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel_xonly, c.x_float_rel, c.type, c.x, c.bin_base);
    }
    else {
      init_aos_xfhmix_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_xonly, c.x_float, c.type, c.x);
    }

    cudaDeviceSynchronize();

    // threadsPerBlock = 128;
    threadsPerBlock = BLOCK_SIZE;
    // threadsPerBlock = 512;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

   if (USE_RELATIVE_COORD) {
      if (NEIGH_STG == NO_NEIGH_SEP) {
        fhmix_basic_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel_xonly, c.x_float_rel,
            c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      }
      else if (NEIGH_STG == BASIC_NEIGH_SEP) {
        fhmix_half2_force_kernel_x_rel_shared_mem<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel_xonly, c.x_float_rel,
            c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

        // fhmix_half2_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel_xonly, c.x_float_rel,
        //     c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);        
      }
      else if (NEIGH_STG == TWO_END_NEIGH) {
        // fhmix_two_end_half2_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numfront, list.d_neighbors, c.x_half_rel_xonly,
        //     __double2half(c.binsizex), __double2half(c.binsizey), __double2half(c.binsizez), f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, __double2half(c.m_cutsq[1][1]));  

        // fhmix_two_end_float_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numback, list.d_neighbors, list.maxneighs, c.x_float_rel,
        //     (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);  

        fhmix_two_end_half2_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numfront, list.d_numback, list.d_neighbors, list.maxneighs, c.x_half_rel_xonly, c.x_float_rel,
            (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);        
      }
      else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
        // fhmix_int2_half2_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
        //     (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

        // fhmix_int2_half2_force_kernel_x_rel_shared_mem<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
        //     (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

        fhmix_expr_optimized<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
            (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);        
      }
      else {
        exit(-1);
      }
    }
   else {
      exit(-1);
    }
    cudaDeviceSynchronize();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    //printf("float kernel launch part3\n"); fflush(stdout);
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    // printf("launch cuda reduce kernel, ntotal : %d, f.extent(0) : %d\n", ntotal, f.extent(0)); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
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
        printf("lazy init x_half_rel_xonly\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_rel_xonly);
        }
        cudaMalloc((void**)&(fpair -> x_half_rel_xonly), (fpair -> x).extent(0) * sizeof(AoS_half_xonly));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_rel_xonly = fpair -> x_half_rel_xonly;
        printf("x_half_rel_xonly extent : %d\n", fpair -> x_half_size);
      }
    }
    else {
      if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
        printf("lazy init x_float\n"); fflush(stdout);
        fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float = fpair -> x_float;
        printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
      }
      if(fpair -> x_half_size < (fpair -> x).extent(0)) {
        printf("lazy init x_half_xonly\n"); fflush(stdout);
        if(fpair -> x_half_size) {
          cudaFree(fpair -> x_half_xonly);
        }
        cudaMalloc((void**)&(fpair -> x_half_xonly), (fpair -> x).extent(0) * sizeof(AoS_half_xonly));
        fpair -> x_half_size = (fpair -> x).extent(0);
        c.x_half_xonly = fpair -> x_half_xonly;
        printf("x_half_xonly extent : %d\n", fpair -> x_half_size);
      }
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
          nmax, c.x_half_rel_xonly, c.x_float_rel, c.type, c.x, c.bin_base);
    }
    else {
      init_aos_xfhmix_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_xonly, c.x_float, c.type, c.x);
    }

    cudaDeviceSynchronize();

    // threadsPerBlock = 128;
    threadsPerBlock = BLOCK_SIZE;
    // threadsPerBlock = 512;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      if (NEIGH_STG == NO_NEIGH_SEP) {
        //exit(-1);
        fhmix_basic_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel_xonly, c.x_float_rel,
            c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      }
      else if (NEIGH_STG == BASIC_NEIGH_SEP) {
        fhmix_half2_force_kernel_x_rel_shared_mem<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel_xonly, c.x_float_rel,
            c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

        // fhmix_half2_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel_xonly, c.x_float_rel,
        //     c.fhcut_split, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      }
      else if (NEIGH_STG == TWO_END_NEIGH) {
        // fhmix_two_end_half2_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numfront, list.d_neighbors, c.x_half_rel_xonly,
        //     __double2half(c.binsizex), __double2half(c.binsizey), __double2half(c.binsizez), f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, __double2half(c.m_cutsq[1][1]));  

        // fhmix_two_end_float_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numback, list.d_neighbors, list.maxneighs, c.x_float_rel,
        //     (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);  

        fhmix_two_end_half2_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numfront, list.d_numback, list.d_neighbors, list.maxneighs, c.x_half_rel_xonly, c.x_float_rel,
            (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      }
      else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
        // fhmix_int2_half2_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
        //     (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

        // fhmix_int2_half2_force_kernel_x_rel_shared_mem<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
        //     ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
        //     (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
        //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

        fhmix_expr_optimized<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
            (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      }
      else {
        exit(-1);
      }
    }
    else {
      exit(-1);
    }
    cudaDeviceSynchronize();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    return ev;
  }
};

template<class PairStyle, unsigned NEIGHFLAG, int ZEROFLAG=0, class Specialisation = void>
EV_FLOAT pair_compute_neighlist_custom (PairStyle* fpair, typename std::enable_if<!((NEIGHFLAG&PairStyle::EnabledNeighFlags) != 0), NeighListKokkos<typename PairStyle::device_type>*>::type list) {
  EV_FLOAT ev;
  (void) fpair;
  (void) list;
  printf("ERROR: calling pair_compute with invalid neighbor list style: requested %i  available %i \n",NEIGHFLAG,PairStyle::EnabledNeighFlags);
  return ev;
}

template<class PairStyle, PRECISION_TYPE PRECTYPE, unsigned NEIGHFLAG, int USE_RELATIVE_COORD, NEIGH_SEP_STRATEGY NEIGH_STG = NO_NEIGH_SEP, int ZEROFLAG = 0, class Specialisation = void>
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

  if (std::is_same<typename DoCoul<PairStyle::COUL_FLAG>::type, CoulTag>::value) {
    printf("ERROR: DoCoul<PairStyle::COUL_FLAG>::type is CoulTag\n");
    exit(1);
  }

  if(PRECTYPE == DOUBLE_PREC) {
    PairComputeFunctorCustomDouble<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD,NEIGH_STG,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      // Kokkos::parallel_reduce(list->inum,ff,ev);
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      //Kokkos::parallel_for(list->inum,ff);
      //ff.test_kernel_launch(list->inum);
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute_custom();
  }
  else if(PRECTYPE == FLOAT_PREC) {
    PairComputeFunctorCustomFloat<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD,NEIGH_STG,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute_custom();
  }
  else if(PRECTYPE == HALF_PREC) {
    PairComputeFunctorCustomHalf<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD,NEIGH_STG,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute_custom();
  }
  else if(PRECTYPE == HFMIX_PREC) {
    PairComputeFunctorCustomFhmix<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD,NEIGH_STG,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
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

  if (fpair->use_relative_coord) {
    if (fpair->neigh_sep_strategy == NO_NEIGH_SEP) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 1, NO_NEIGH_SEP, 1, Specialisation> (fpair,list);
    }
    else if (fpair->neigh_sep_strategy == BASIC_NEIGH_SEP) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 1, BASIC_NEIGH_SEP, 1, Specialisation> (fpair,list);
    }
    else if (fpair->neigh_sep_strategy == TWO_END_NEIGH) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 1, TWO_END_NEIGH, 1, Specialisation> (fpair,list);
    }
    else if (fpair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 1, TWO_END_NEIGH_INT2, 1, Specialisation> (fpair,list);
    }
    else {
      printf("ERROR: NEIGH_SEP_STRATEGY not implemented\n");
      exit(1);
    }
  }
  else {
    if (fpair->neigh_sep_strategy == NO_NEIGH_SEP) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 0, NO_NEIGH_SEP, 1, Specialisation> (fpair,list);
    }
    else if (fpair->neigh_sep_strategy == BASIC_NEIGH_SEP) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 0, BASIC_NEIGH_SEP, 1, Specialisation> (fpair,list);
    }
    else if (fpair->neigh_sep_strategy == TWO_END_NEIGH) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 0, TWO_END_NEIGH, 1, Specialisation> (fpair,list);
    }
    else if (fpair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
      ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, FULL, 0, TWO_END_NEIGH_INT2, 1, Specialisation> (fpair,list);
    }
    else {
      printf("ERROR: NEIGH_SEP_STRATEGY not implemented\n");
      exit(1);
    }    
  }
  return ev;
}

} // namespace LJKernels
