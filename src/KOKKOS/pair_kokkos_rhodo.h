namespace RhodoKernels {

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void double_force_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  KKScatterView<F_FLOAT*[3], typename ArrayTypes<DeviceType>::t_f_array::array_layout,typename KKDevice<DeviceType>::value,KKScatterSum,typename NeedDup<NEIGHFLAG,DeviceType>::value> dup_f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double cut_coulsq, double cut_coul_innersq, double denom_coul) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

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

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = 1.0;
      const F_FLOAT factor_coul = 1.0;
      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < d_cut_ljsq(itype,jtype)) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (params(itype,jtype).lj1*r6inv -
            params(itype,jtype).lj2);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            switch2 = 12.0*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            englj = r6inv *
                    (params(itype,jtype).lj3*r6inv -
                    params(itype,jtype).lj4);
            forcelj = forcelj*switch1 + englj*switch2;
          }

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul, switch1;

          forcecoul = qqrd2e * qtmp * q(j) *rinv;

          if (rsq > cut_coul_innersq) {
            switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                      (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) / denom_coul;
            forcecoul *= switch1;
          }

          fpair += forcecoul * r2inv * factor_coul;

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        //if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        //}

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              //evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              F_FLOAT englj, switch1;

              englj = r6inv *
                (params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj4);

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0*evdwl;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              // ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);
              F_FLOAT switch1;

              ecoul = qqrd2e * qtmp * q(j) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0*ecoul;
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
                ev.v[0] += v0;
                ev.v[1] += v1;
                ev.v[2] += v2;
                ev.v[3] += v3;
                ev.v[4] += v4;
                ev.v[5] += v5;
            }
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    a_f(i,0) += fxtmp;
    a_f(i,1) += fytmp;
    a_f(i,2) += fztmp;
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void double_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x_rel, X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  KKScatterView<F_FLOAT*[3], typename ArrayTypes<DeviceType>::t_f_array::array_layout,typename KKDevice<DeviceType>::value,KKScatterSum,typename NeedDup<NEIGHFLAG,DeviceType>::value> dup_f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double cut_coulsq, double cut_coul_innersq, double denom_coul) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

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

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < d_cut_ljsq(itype,jtype)) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (params(itype,jtype).lj1*r6inv -
            params(itype,jtype).lj2);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            switch2 = 12.0*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            englj = r6inv *
                    (params(itype,jtype).lj3*r6inv -
                    params(itype,jtype).lj4);
            forcelj = forcelj*switch1 + englj*switch2;
          }

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul, switch1;

          forcecoul = qqrd2e * qtmp * q(j) *rinv;

          if (rsq > cut_coul_innersq) {
            switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                      (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) / denom_coul;
            forcecoul *= switch1;
          }

          fpair += forcecoul * r2inv * factor_coul;

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        //if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        //}

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              F_FLOAT englj, switch1;

              englj = r6inv *
                (params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj4);

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0*evdwl;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);
              F_FLOAT switch1;

              ecoul = qqrd2e * qtmp * q(j) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0*ecoul;
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
                ev.v[0] += v0;
                ev.v[1] += v1;
                ev.v[2] += v2;
                ev.v[3] += v3;
                ev.v[4] += v4;
                ev.v[5] += v5;
            }
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    a_f(i,0) += fxtmp;
    a_f(i,1) += fytmp;
    a_f(i,2) += fztmp;
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void double_force_kernel_x_rel_f_atomic(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x_rel, X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double cut_coulsq, double cut_coul_innersq, double denom_coul) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x_rel(i,0);
    const X_FLOAT ytmp = x_rel(i,1);
    const X_FLOAT ztmp = x_rel(i,2);
    const int itype = type(i);
    const F_FLOAT qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

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

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < d_cut_ljsq(itype,jtype)) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (params(itype,jtype).lj1*r6inv -
            params(itype,jtype).lj2);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            switch2 = 12.0*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            englj = r6inv *
                    (params(itype,jtype).lj3*r6inv -
                    params(itype,jtype).lj4);
            forcelj = forcelj*switch1 + englj*switch2;
          }

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul, switch1;

          forcecoul = qqrd2e * qtmp * q(j) *rinv;

          if (rsq > cut_coul_innersq) {
            switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                      (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) / denom_coul;
            forcecoul *= switch1;
          }

          fpair += forcecoul * r2inv * factor_coul;

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));
        // f_ptr[j * 3 + 0] += (double)(-delx*fpair);
        // f_ptr[j * 3 + 1] += (double)(-dely*fpair);
        // f_ptr[j * 3 + 2] += (double)(-delz*fpair);

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              F_FLOAT englj, switch1;

              englj = r6inv *
                (params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj4);

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0*evdwl;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);
              F_FLOAT switch1;

              ecoul = qqrd2e * qtmp * q(j) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0*ecoul;
            }
          }

          // if (vflag_either) {
          //   const E_FLOAT v0 = delx*delx*fpair;
          //   const E_FLOAT v1 = dely*dely*fpair;
          //   const E_FLOAT v2 = delz*delz*fpair;
          //   const E_FLOAT v3 = delx*dely*fpair;
          //   const E_FLOAT v4 = delx*delz*fpair;
          //   const E_FLOAT v5 = dely*delz*fpair;

          //   if (vflag_global) {
          //       ev.v[0] += v0;
          //       ev.v[1] += v1;
          //       ev.v[2] += v2;
          //       ev.v[3] += v3;
          //       ev.v[4] += v4;
          //       ev.v[5] += v5;
          //   }
          // }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
    // f_ptr[i * 3 + 0] += (double)fxtmp;
    // f_ptr[i * 3 + 1] += (double)fytmp;
    // f_ptr[i * 3 + 2] += (double)fztmp;
  }
}

// use fhcut_split as charmm border 
template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void double_force_kernel_x_rel_fhcut_split(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x_rel, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread fhcut_split,
  X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double cut_coulsq, double cut_coul_innersq, double denom_coul) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x_rel(i,0);
    const X_FLOAT ytmp = x_rel(i,1);
    const X_FLOAT ztmp = x_rel(i,2);
    const int itype = type(i);
    const F_FLOAT qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int fh_cut = fhcut_split(i);
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    int jj;
    for (jj = 0; jj < fh_cut; jj++) {
      int ni = neighbors_i(jj);
      const F_FLOAT factor_lj = 1.0;
      const F_FLOAT factor_coul = 1.0;
      int j = ni & DIRNEIGHMASK;
      const X_FLOAT delx = xtmp - x_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const X_FLOAT dely = ytmp - x_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const X_FLOAT delz = ztmp - x_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < d_cut_ljsq(itype,jtype)) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (params(itype,jtype).lj1*r6inv -
            params(itype,jtype).lj2);

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul, switch1;

          forcecoul = qqrd2e * qtmp * q(j) *rinv;

          fpair += forcecoul * r2inv * factor_coul;
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              F_FLOAT englj, switch1;

              englj = r6inv *
                (params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj4);

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0*evdwl;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);
              F_FLOAT switch1;

              ecoul = qqrd2e * qtmp * q(j) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0*ecoul;
            }
          }
        }
      }
    }
    for (; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const F_FLOAT factor_lj = 1.0;
      const F_FLOAT factor_coul = 1.0;
      int j = ni & DIRNEIGHMASK;
      const X_FLOAT delx = xtmp - x_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
      const X_FLOAT dely = ytmp - x_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
      const X_FLOAT delz = ztmp - x_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < d_cut_ljsq(itype,jtype)) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (params(itype,jtype).lj1*r6inv -
            params(itype,jtype).lj2);

          switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                    (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
          switch2 = 12.0*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
          englj = r6inv *
                  (params(itype,jtype).lj3*r6inv -
                  params(itype,jtype).lj4);
          forcelj = forcelj*switch1 + englj*switch2;

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul, switch1;

          forcecoul = qqrd2e * qtmp * q(j) *rinv;

          switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                    (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) / denom_coul;
          forcecoul *= switch1;

          fpair += forcecoul * r2inv * factor_coul;
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              F_FLOAT englj, switch1;

              englj = r6inv *
                (params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj4);

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0*evdwl;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);
              F_FLOAT switch1;

              ecoul = qqrd2e * qtmp * q(j) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0*ecoul;
            }
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
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

    // if (NEIGHFLAG == FULL && ZEROFLAG) {
    //   f(i,0) = 0.0;
    //   f(i,1) = 0.0;
    //   f(i,2) = 0.0;
    // }

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      // const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      // const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
      // j &= NEIGHMASK;
      const F_FLOAT factor_lj = 1.0;
      const F_FLOAT factor_coul = 1.0;
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < c.d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < c.d_cut_ljsq(itype,jtype)) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (c.params(itype,jtype).lj1*r6inv -
            c.params(itype,jtype).lj2);

          if (rsq > c.cut_lj_innersq) {
            switch1 = (c.cut_ljsq-rsq) * (c.cut_ljsq-rsq) *
                      (c.cut_ljsq + 2.0*rsq - 3.0*c.cut_lj_innersq) / c.denom_lj;
            switch2 = 12.0*rsq * (c.cut_ljsq-rsq) * (rsq-c.cut_lj_innersq) / c.denom_lj;
            englj = r6inv *
                    (c.params(itype,jtype).lj3*r6inv -
                    c.params(itype,jtype).lj4);
            forcelj = forcelj*switch1 + englj*switch2;
          }

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < c.d_cut_coulsq(itype,jtype)) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT rinv = sqrt(r2inv);
          F_FLOAT forcecoul, switch1;

          forcecoul = c.qqrd2e*qtmp*c.q(j) *rinv;

          if (rsq > c.cut_coul_innersq) {
            switch1 = (c.cut_coulsq-rsq) * (c.cut_coulsq-rsq) *
                      (c.cut_coulsq + 2.0*rsq - 3.0*c.cut_coul_innersq) / c.denom_coul;
            forcecoul *= switch1;
          }

          fpair += forcecoul * r2inv * factor_coul;

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        //if ((NEIGHFLAG==HALF || NEIGHFLAG==HALFTHREAD) && (NEWTON_PAIR || j < c.nlocal)) {
          a_f(j,0) -= delx*fpair;
          a_f(j,1) -= dely*fpair;
          a_f(j,2) -= delz*fpair;
        //}

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag) {
            if (rsq < c.d_cut_ljsq(itype,jtype)) {
              //evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              F_FLOAT englj, switch1;

              englj = r6inv *
                (c.params(itype,jtype).lj3*r6inv -
                c.params(itype,jtype).lj4);

              if (rsq > c.cut_lj_innersq) {
                switch1 = (c.cut_ljsq-rsq) * (c.cut_ljsq-rsq) *
                  (c.cut_ljsq + 2.0*rsq - 3.0*c.cut_lj_innersq) / c.denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0*evdwl;
            }
            if (rsq < c.d_cut_coulsq(itype,jtype)) {
              // ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);
              F_FLOAT switch1;

              ecoul = c.qqrd2e * qtmp * c.q(j) * rinv;
              if (rsq > c.cut_coul_innersq) {
                switch1 = (c.cut_coulsq-rsq) * (c.cut_coulsq-rsq) *
                          (c.cut_coulsq + 2.0*rsq - 3.0*c.cut_coul_innersq) /
                          c.denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0*ecoul;
            }
          }

          if (c.vflag_either || c.eflag_atom) {
            //ev_tally(ev,i,j,evdwl+ecoul,fpair,delx,dely,delz);
            // auto a_eatom = dup_eatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();
            // auto a_vatom = dup_vatom.template access<typename AtomicDup<NEIGHFLAG,device_type>::value>();

            const int VFLAG = c.vflag_either;

            if (VFLAG) {
              const E_FLOAT v0 = delx*delx*fpair;
              const E_FLOAT v1 = dely*dely*fpair;
              const E_FLOAT v2 = delz*delz*fpair;
              const E_FLOAT v3 = delx*dely*fpair;
              const E_FLOAT v4 = delx*delz*fpair;
              const E_FLOAT v5 = dely*delz*fpair;

              if (c.vflag_global) {
                  ev.v[0] += v0;
                  ev.v[1] += v1;
                  ev.v[2] += v2;
                  ev.v[3] += v3;
                  ev.v[4] += v4;
                  ev.v[5] += v5;
              }
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
            ev.v[0] += v0;
            ev.v[1] += v1;
            ev.v[2] += v2;
            ev.v[3] += v3;
            ev.v[4] += v4;
            ev.v[5] += v5;
      }

      /*if (c.vflag_atom) {
          a_vatom(i,0) += 0.5*v0;
          a_vatom(i,1) += 0.5*v1;
          a_vatom(i,2) += 0.5*v2;
          a_vatom(i,3) += 0.5*v3;
          a_vatom(i,4) += 0.5*v4;
          a_vatom(i,5) += 0.5*v5;

          a_vatom(j,0) += 0.5*v0;
          a_vatom(j,1) += 0.5*v1;
          a_vatom(j,2) += 0.5*v2;
          a_vatom(j,3) += 0.5*v3;
          a_vatom(j,4) += 0.5*v4;
          a_vatom(j,5) += 0.5*v5;
      }*/
    }
  }


  KOKKOS_INLINE_FUNCTION
  void operator()(const int i) const {
    //if (c.newton_pair) 
    compute_item_custom<0,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    //else compute_item_custom<0,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &energy_virial) const {
    //if (c.newton_pair)
    energy_virial += compute_item_custom<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    //else
    //  energy_virial += compute_item_custom<1,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
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
      // double_kernel_x_rel_check<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.x, c.binsizex, c.binsizey, c.binsizez, c.type);

      // double_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
      //     c.type, c.q, dup_f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
      // printf("in normal kernel_launch\n");

      double_force_kernel_x_rel_f_atomic<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);

      // double_force_kernel_x_rel_fhcut_split<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, 
      //     c.fhcut_split, c.binsizex, c.binsizey, c.binsizez,
      //     c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
    }
    else {
      double_force_kernel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, c.q, dup_f,
          c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
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
      // double_kernel_x_rel_check<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.x, c.binsizex, c.binsizey, c.binsizez, c.type);

      // double_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
      //     c.type, c.q, dup_f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
      // printf("in reduce kernel\n");

      double_force_kernel_x_rel_f_atomic<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);

      // double_force_kernel_x_rel_fhcut_split<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_rel, 
      //     c.fhcut_split, c.binsizex, c.binsizey, c.binsizez,
      //     c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
    }
    else {
      double_force_kernel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, c.q, dup_f,
          c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
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

// use custom atomic by default
template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel, float binsizex, float binsizey, float binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float cut_coulsq, float cut_coul_innersq, float denom_coul) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float_rel(i,0);
    const float ytmp = x_float_rel(i,1);
    const float ztmp = x_float_rel(i,2);
    const int itype = type(i);
    const float qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

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

      if (rsq < (float)d_cutsq(itype,jtype)) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (((float)params(itype,jtype).lj1)*r6inv -
            ((float)params(itype,jtype).lj2));

          if (rsq > (float)cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            englj = r6inv *
                    (((float)params(itype,jtype).lj3)*r6inv -
                    ((float)params(itype,jtype).lj4));
            forcelj = forcelj*switch1 + englj*switch2;
          }

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < (float)(d_cut_coulsq(itype,jtype))) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const float r2inv = 1.0f/rsq;
          const float rinv = sqrtf(r2inv);
          float forcecoul, switch1;

          forcecoul = qqrd2e * qtmp * (float)(q(j)) *rinv;

          if (rsq > cut_coul_innersq) {
            switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                      (cut_coulsq + 2.0f*rsq - 3.0f*cut_coul_innersq) / denom_coul;
            forcecoul *= switch1;
          }

          fpair += forcecoul * r2inv * factor_coul;

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < (float)(d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              float englj, switch1;

              englj = r6inv *
                (((float)params(itype,jtype).lj3)*r6inv -
                ((float)params(itype,jtype).lj4));

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < (float)(d_cut_coulsq(itype,jtype))) {
              const float r2inv = 1.0f/rsq;
              const float rinv = sqrtf(r2inv);
              float switch1;

              ecoul = qqrd2e * qtmp * ((float)q(j)) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0f*rsq - 3.0f*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0f*ecoul;
            }
          }

          // if (vflag_either) {
          //   const E_FLOAT v0 = delx*delx*fpair;
          //   const E_FLOAT v1 = dely*dely*fpair;
          //   const E_FLOAT v2 = delz*delz*fpair;
          //   const E_FLOAT v3 = delx*dely*fpair;
          //   const E_FLOAT v4 = delx*delz*fpair;
          //   const E_FLOAT v5 = dely*delz*fpair;

          //   if (vflag_global) {
          //       ev.v[0] += v0;
          //       ev.v[1] += v1;
          //       ev.v[2] += v2;
          //       ev.v[3] += v3;
          //       ev.v[4] += v4;
          //       ev.v[5] += v5;
          //   }
          // }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
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

  PairComputeFunctorCustomFloat(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomFloat() {c.copymode = 1; list.copymode = 1;};

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

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated) {
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
      float_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
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
      float_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
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

// use custom atomic by default
template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void half_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, __half binsizex_h, __half binsizey_h, __half binsizez_h, 
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  __half qqrd2e_h, __half cut_ljsq_h, __half cut_lj_innersq_h, __half denom_lj_h, __half cut_coulsq_h, __half cut_coul_innersq_h, __half denom_coul_h) {

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
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    __half fxtmp = __float2half(0.0f);
    __half fytmp = __float2half(0.0f);
    __half fztmp = __float2half(0.0f);

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const __half factor_lj = __float2half(1.0f);
      const __half factor_coul = __float2half(1.0f);
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx = xtmp - half_data_j.x[0] - __int2half_rn(((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * binsizex_h;
      const __half dely = ytmp - half_data_j.x[1] - __int2half_rn(((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * binsizey_h;
      const __half delz = ztmp - half_data_j.x[2] - __int2half_rn(((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * binsizez_h;

      const int jtype = half_data_j.type;
      const __half rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < __double2half(d_cutsq(itype,jtype))) {

        __half fpair = __float2half(0.0f);

        if (rsq < __double2half(d_cut_ljsq(itype,jtype))) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const __half r2inv = __float2half(1.0f)/rsq;
          const __half r6inv = r2inv*r2inv*r2inv;
          __half forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (__double2half(params(itype,jtype).lj1)*r6inv -
            __double2half(params(itype,jtype).lj2));

          if (rsq > cut_lj_innersq_h) {
            switch1 = (cut_ljsq_h-rsq) * (cut_ljsq_h-rsq) *
                      (cut_ljsq_h + __float2half(2.0f)*rsq - __float2half(3.0f)*cut_lj_innersq_h) / denom_lj_h;
            switch2 = __float2half(12.0f)*rsq * (cut_ljsq_h-rsq) * (rsq-cut_lj_innersq_h) / denom_lj_h;
            englj = r6inv *
                    (__double2half(params(itype,jtype).lj3)*r6inv -
                    __double2half(params(itype,jtype).lj4));
            forcelj = forcelj*switch1 + englj*switch2;
          }

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < __double2half(d_cut_coulsq(itype,jtype))) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const __half r2inv = __float2half(1.0f)/rsq;
          const __half rinv = hsqrt(r2inv);
          __half forcecoul, switch1;

          forcecoul = qqrd2e_h * qtmp * __double2half(q(j)) *rinv;

          if (rsq > cut_coul_innersq_h) {
            switch1 = (cut_coulsq_h-rsq) * (cut_coulsq_h-rsq) *
                      (cut_coulsq_h + __float2half(2.0f)*rsq - __float2half(3.0f)*cut_coul_innersq_h) / denom_coul_h;
            forcecoul *= switch1;
          }

          fpair += forcecoul * r2inv * factor_coul;

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_ptr[j * 3 + 0], (double)(__half2float(-delx*fpair)));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(__half2float(-dely*fpair)));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(__half2float(-delz*fpair)));

        if (EVFLAG) {
          __half evdwl = __float2half(0.0f);
          __half ecoul = __float2half(0.0f);
          if (eflag) {
            if (rsq < __double2half(d_cut_ljsq(itype,jtype))) {
              const __half r2inv = __float2half(1.0f)/rsq;
              const __half r6inv = r2inv*r2inv*r2inv;
              __half englj, switch1;

              englj = r6inv *
                (__double2half(params(itype,jtype).lj3)*r6inv -
                __double2half(params(itype,jtype).lj4));

              if (rsq > cut_lj_innersq_h) {
                switch1 = (cut_ljsq_h-rsq) * (cut_ljsq_h-rsq) *
                  (cut_ljsq_h + __float2half(2.0f)*rsq - __float2half(3.0f)*cut_lj_innersq_h) / denom_lj_h;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0f*__half2float(evdwl);
            }
            if (rsq < __double2half(d_cut_coulsq(itype,jtype))) {
              const __half r2inv = __float2half(1.0f)/rsq;
              const __half rinv = hsqrt(r2inv);
              __half switch1;

              ecoul = qqrd2e_h * qtmp * __double2half(q(j)) * rinv;
              if (rsq > cut_coul_innersq_h) {
                switch1 = (cut_coulsq_h-rsq) * (cut_coulsq_h-rsq) *
                          (cut_coulsq_h + __float2half(2.0f)*rsq - __float2half(3.0f)*cut_coul_innersq_h) /
                          denom_coul_h;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0f* __half2float(ecoul);
            }
          }

          // if (vflag_either) {
          //   const E_FLOAT v0 = delx*delx*fpair;
          //   const E_FLOAT v1 = dely*dely*fpair;
          //   const E_FLOAT v2 = delz*delz*fpair;
          //   const E_FLOAT v3 = delx*dely*fpair;
          //   const E_FLOAT v4 = delx*delz*fpair;
          //   const E_FLOAT v5 = dely*delz*fpair;

          //   if (vflag_global) {
          //       ev.v[0] += v0;
          //       ev.v[1] += v1;
          //       ev.v[2] += v2;
          //       ev.v[3] += v3;
          //       ev.v[4] += v4;
          //       ev.v[5] += v5;
          //   }
          // }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    atomicAdd(&f_ptr[i * 3 + 0], (double)__half2float(fxtmp));
    atomicAdd(&f_ptr[i * 3 + 1], (double)__half2float(fytmp));
    atomicAdd(&f_ptr[i * 3 + 2], (double)__half2float(fztmp));
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void half_force_kernel_x_rel_sim_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, __half binsizex_h, __half binsizey_h, __half binsizez_h, 
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  __half qqrd2e_h, __half cut_ljsq_h, __half cut_lj_innersq_h, __half denom_lj_h, __half cut_coulsq_h, __half cut_coul_innersq_h, __half denom_coul_h,
  double qqrd2e_d, double cut_ljsq_d, double cut_lj_innersq_d, double denom_lj_d, double cut_coulsq_d, double cut_coul_innersq_d, double denom_coul_d) {

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
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    __half fxtmp = __float2half(0.0f);
    __half fytmp = __float2half(0.0f);
    __half fztmp = __float2half(0.0f);

    for (int jj = 0; jj < jnum; jj++) {
      int ni = neighbors_i(jj);
      const __half factor_lj = __float2half(1.0f);
      const __half factor_coul = __float2half(1.0f);
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx = xtmp - half_data_j.x[0] - __int2half_rn(((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * binsizex_h;
      const __half dely = ytmp - half_data_j.x[1] - __int2half_rn(((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * binsizey_h;
      const __half delz = ztmp - half_data_j.x[2] - __int2half_rn(((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * binsizez_h;

      const int jtype = half_data_j.type;
      const __half rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < __double2half(d_cutsq(itype,jtype))) {

        __half fpair = __float2half(0.0f);

        if (rsq < __double2half(d_cut_ljsq(itype,jtype))) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const __half r2inv = __float2half(1.0f)/rsq;
          const __half r6inv = r2inv*r2inv*r2inv;
          __half forcelj, englj;

          forcelj = r6inv *
            (__double2half(params(itype,jtype).lj1)*r6inv -
            __double2half(params(itype,jtype).lj2));

          if (rsq > cut_lj_innersq_h) {
            double rsq_d = (double)__half2float(rsq);
            double switch1_d = (cut_ljsq_d-rsq_d) * (cut_ljsq_d-rsq_d) *
                      (cut_ljsq_d + 2.0*rsq_d - 3.0*cut_lj_innersq_d) / denom_lj_d;
            double switch2_d = 12.0*rsq_d * (cut_ljsq_d-rsq_d) * (rsq_d-cut_lj_innersq_d) / denom_lj_d;
            englj = r6inv *
                    (__double2half(params(itype,jtype).lj3)*r6inv -
                    __double2half(params(itype,jtype).lj4));
            forcelj = forcelj*__double2half(switch1_d) + englj*__double2half(switch2_d);
          }

          fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < __double2half(d_cut_coulsq(itype,jtype))) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          const __half r2inv = __float2half(1.0f)/rsq;
          const __half rinv = hsqrt(r2inv);
          __half forcecoul;

          forcecoul = qqrd2e_h * qtmp * __double2half(q(j)) *rinv;

          if (rsq > cut_coul_innersq_h) {
            double rsq_d = (double)__half2float(rsq);
            double switch1_d = (cut_coulsq_d-rsq_d) * (cut_coulsq_d-rsq_d) *
                      (cut_coulsq_d + 2.0*rsq_d - 3.0*cut_coul_innersq_d) / denom_coul_d;
            forcecoul *= __double2half(switch1_d);
          }

          fpair += forcecoul * r2inv * factor_coul;

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_ptr[j * 3 + 0], (double)(__half2float(-delx*fpair)));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(__half2float(-dely*fpair)));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(__half2float(-delz*fpair)));

        if (EVFLAG) {
          double evdwl_d = 0.0;
          double ecoul_d = 0.0;
          if (eflag) {
            if (rsq < __double2half(d_cut_ljsq(itype,jtype))) {
              double rsq_d = (double)__half2float(rsq);
              const double r2inv_d = 1.0/rsq_d;
              const double r6inv_d = r2inv_d*r2inv_d*r2inv_d;
              double englj_d, switch1_d;

              englj_d = r6inv_d *
                ((params(itype,jtype).lj3)*r6inv_d -
                (params(itype,jtype).lj4));

              if (rsq_d > cut_lj_innersq_d) {
                switch1_d = (cut_ljsq_d-rsq_d) * (cut_ljsq_d-rsq_d) *
                  (cut_ljsq_d + 2.0*rsq_d - 3.0*cut_lj_innersq_d) / denom_lj_d;
                englj_d *= switch1_d;
              }

              evdwl_d = __half2float(factor_lj) * englj_d;
              ev.evdwl += 1.0*evdwl_d;
            }
            if (rsq < __double2half(d_cut_coulsq(itype,jtype))) {
              double rsq_d = (double)__half2float(rsq);
              const double r2inv_d = 1.0/rsq_d;
              const double rinv_d = sqrt(r2inv_d);
              double switch1_d;

              ecoul_d = qqrd2e_d * __half2float(qtmp) * q(j) * rinv_d;
              if (rsq_d > cut_coul_innersq_d) {
                switch1_d = (cut_coulsq_d-rsq_d) * (cut_coulsq_d-rsq_d) *
                          (cut_coulsq_d + 2.0*rsq_d - 3.0*cut_coul_innersq_d) /
                          denom_coul_d;
                ecoul_d *= switch1_d;
              }

              ecoul_d *= __half2float(factor_coul);

              ev.ecoul += 1.0 * ecoul_d;
            }
          }

          // if (vflag_either) {
          //   const E_FLOAT v0 = delx*delx*fpair;
          //   const E_FLOAT v1 = dely*dely*fpair;
          //   const E_FLOAT v2 = delz*delz*fpair;
          //   const E_FLOAT v3 = delx*dely*fpair;
          //   const E_FLOAT v4 = delx*delz*fpair;
          //   const E_FLOAT v5 = dely*delz*fpair;

          //   if (vflag_global) {
          //       ev.v[0] += v0;
          //       ev.v[1] += v1;
          //       ev.v[2] += v2;
          //       ev.v[3] += v3;
          //       ev.v[4] += v4;
          //       ev.v[5] += v5;
          //   }
          // }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    atomicAdd(&f_ptr[i * 3 + 0], (double)__half2float(fxtmp));
    atomicAdd(&f_ptr[i * 3 + 1], (double)__half2float(fytmp));
    atomicAdd(&f_ptr[i * 3 + 2], (double)__half2float(fztmp));
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

  PairComputeFunctorCustomHalf(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomHalf() {c.copymode = 1; list.copymode = 1;};

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

#ifdef RELATIVE_COORD
    // half_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
    //     c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     c.qqrd2e_h, c.cut_ljsq_h, c.cut_lj_innersq_h, c.denom_lj_h, c.cut_coulsq_h, c.cut_coul_innersq_h, c.denom_coul_h);
    half_force_kernel_x_rel_sim_table<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
        ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
        c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
        c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
        c.qqrd2e_h, c.cut_ljsq_h, c.cut_lj_innersq_h, c.denom_lj_h, c.cut_coulsq_h, c.cut_coul_innersq_h, c.denom_coul_h,
        c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
#endif
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
      // half_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
      //     c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     c.qqrd2e_h, c.cut_ljsq_h, c.cut_lj_innersq_h, c.denom_lj_h, c.cut_coulsq_h, c.cut_coul_innersq_h, c.denom_coul_h);
      half_force_kernel_x_rel_sim_table<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.binsizex_h, c.binsizey_h, c.binsizez_h,
          c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e_h, c.cut_ljsq_h, c.cut_lj_innersq_h, c.denom_lj_h, c.cut_coulsq_h, c.cut_coul_innersq_h, c.denom_coul_h,
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
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

// use custom atomic by default
template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void fhmix_force_kernel_x_rel_split_cut_inner(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_half* x_half_rel, Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel, 
  float binsizex, float binsizey, float binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  __half qqrd2e_h, __half cut_ljsq_h, __half cut_lj_innersq_h, __half denom_lj_h, __half cut_coulsq_h, __half cut_coul_innersq_h, __half denom_coul_h,
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float cut_coulsq, float cut_coul_innersq, float denom_coul) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float_rel(i,0);
    const float ytmp = x_float_rel(i,1);
    const float ztmp = x_float_rel(i,2);

    const AoS_half half_data = x_half_rel[i];
    const __half xtmp_h = half_data.x[0];
    const __half ytmp_h = half_data.x[1];
    const __half ztmp_h = half_data.x[2];

    const int itype = type(i);
    const float qtmp = q(i);
    const __half qtmp_h = __double2half(q(i));
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    __half fxtmp_h = __float2half(0.0f);
    __half fytmp_h = __float2half(0.0f);
    __half fztmp_h = __float2half(0.0f);

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

      const __half factor_lj_h = __float2half(1.0f);
      const __half factor_coul_h = __float2half(1.0f);
      const AoS_half half_data_j = x_half_rel[j];
      const __half delx_h = xtmp_h - half_data_j.x[0] - __int2half_rn((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * __float2half(binsizex);
      const __half dely_h = ytmp_h - half_data_j.x[1] - __int2half_rn((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * __float2half(binsizey);
      const __half delz_h = ztmp_h - half_data_j.x[2] - __int2half_rn((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * __float2half(binsizez);
      const __half rsq_h = delx_h*delx_h + dely_h*dely_h + delz_h*delz_h;

      if (rsq < (float)d_cutsq(itype,jtype)) {
        float fpair = 0.0f;
        __half fpair_h = __float2half(0.0f);

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          if (rsq > (float)cut_lj_innersq) {
            /// total half
            // const half r2inv_h = __float2half(1.0f)/rsq_h;
            // const half r6inv_h = r2inv_h*r2inv_h*r2inv_h;
            // __half forcelj_h, switch1_h, switch2_h, englj_h;

            // forcelj_h = r6inv_h *
            //   (__float2half((float)params(itype,jtype).lj1)*r6inv_h -
            //   __float2half((float)params(itype,jtype).lj2));

            // switch1_h = (cut_ljsq_h-rsq_h) * (cut_ljsq_h-rsq_h) *
            //           (cut_ljsq_h + __float2half(2.0f)*rsq_h - __float2half(3.0f)*cut_lj_innersq_h) / denom_lj_h;
            // switch2_h = __float2half(12.0f)*rsq_h * (cut_ljsq_h-rsq_h) * (rsq_h-cut_lj_innersq_h) / denom_lj_h;
            // englj_h = r6inv_h *
            //         (__float2half((float)params(itype,jtype).lj3)*r6inv_h -
            //         __float2half((float)params(itype,jtype).lj4));
            // forcelj_h = forcelj_h*switch1_h + englj_h*switch2_h;

            // fpair_h+=factor_lj_h*forcelj_h*r2inv_h;


            /// half with half table
            // const half r2inv_h = __float2half(1.0f)/rsq_h;
            // const half r6inv_h = r2inv_h*r2inv_h*r2inv_h;
            // __half forcelj_h, englj_h;

            // forcelj_h = r6inv_h *
            //   (__float2half((float)params(itype,jtype).lj1)*r6inv_h -
            //   __float2half((float)params(itype,jtype).lj2));

            // double rsq_d = (double)__half2float(rsq_h);              
            // double switch1_d = (cut_ljsq-rsq_d) * (cut_ljsq-rsq_d) *
            //           (cut_ljsq + 2.0*rsq_d - 3.0*cut_lj_innersq) / denom_lj;
            // double switch2_d = 12.0*rsq_d * (cut_ljsq-rsq_d) * (rsq_d-cut_lj_innersq) / denom_lj;
            // englj_h = r6inv_h *
            //         (__float2half((float)params(itype,jtype).lj3)*r6inv_h -
            //         __float2half((float)params(itype,jtype).lj4));
            // forcelj_h = forcelj_h*__double2half(switch1_d) + englj_h*__double2half(switch2_d);

            // fpair_h+=factor_lj_h*forcelj_h*r2inv_h;


            /// float with half table
            // const float r2inv = 1.0f/__half2float(rsq_h);
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // double rsq_d = (double)__half2float(rsq_h);              
            // double switch1_d = (cut_ljsq-rsq_d) * (cut_ljsq-rsq_d) *
            //           (cut_ljsq + 2.0*rsq_d - 3.0*cut_lj_innersq) / denom_lj;
            // double switch2_d = 12.0*rsq_d * (cut_ljsq-rsq_d) * (rsq_d-cut_lj_innersq) / denom_lj;
            // englj = r6inv *
            //         (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * ((float)switch1_d) + englj * ((float)switch2_d);

            // fpair_h+=__float2half(factor_lj*forcelj*r2inv);


            /// float with half without table
            // const float r2inv = 1.0f/__half2float(rsq_h);
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, englj;
            // __half switch1_h, switch2_h;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));
            
            // switch1_h = (cut_ljsq_h - rsq_h) * (cut_ljsq_h - rsq_h) *
            //           (cut_ljsq_h + __float2half(2.0f) * rsq_h - __float2half(3.0f) * cut_lj_innersq_h) / denom_lj_h;
            // switch2_h = __float2half(12.0f) * rsq_h * (cut_ljsq_h - rsq_h) * (rsq_h - cut_lj_innersq_h) / denom_lj_h;
            // englj = r6inv *
            //         (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(switch1_h) + englj * __half2float(switch2_h);

            // fpair_h+=__float2half(factor_lj*forcelj*r2inv);


            /// total float
            const float r2inv = 1.0f/rsq;
            const float r6inv = r2inv*r2inv*r2inv;
            float forcelj, switch1, switch2, englj;

            forcelj = r6inv *
              (((float)params(itype,jtype).lj1)*r6inv -
              ((float)params(itype,jtype).lj2));

            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            englj = r6inv *
                    (((float)params(itype,jtype).lj3)*r6inv -
                    ((float)params(itype,jtype).lj4));
            forcelj = forcelj*switch1 + englj*switch2;

            fpair+=factor_lj*forcelj*r2inv;            
            //fpair_h+= __float2half(factor_lj*forcelj*r2inv); 
          }
          else {
            const float r2inv = 1.0f/rsq;
            const float r6inv = r2inv*r2inv*r2inv;
            float forcelj, switch1, switch2, englj;

            forcelj = r6inv *
              (((float)params(itype,jtype).lj1)*r6inv -
              ((float)params(itype,jtype).lj2));

            fpair+=factor_lj*forcelj*r2inv;
          }

        }
        if (rsq < (float)(d_cut_coulsq(itype,jtype))) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

          if (rsq > cut_coul_innersq) {
            /// total half
            // const __half r2inv_h = __float2half(1.0f)/rsq_h;
            // const __half rinv_h = hsqrt(r2inv_h);
            // __half forcecoul_h, switch1_h;
            // forcecoul_h = qqrd2e_h * qtmp_h * __float2half((float)(q(j))) *rinv_h;
            // switch1_h = (cut_coulsq_h - rsq_h) * (cut_coulsq_h - rsq_h) *
            //           (cut_coulsq_h + __float2half(2.0f) * rsq_h - __float2half(3.0f) * cut_coul_innersq_h) / denom_coul_h;
            // forcecoul_h *= switch1_h;
            // fpair_h += forcecoul_h * r2inv_h * factor_coul_h;


            /// half with half table
            double rsq_d = (double)__half2float(rsq_h);
            double switch1_d = (cut_coulsq-rsq_d) * (cut_coulsq-rsq_d) *
                      (cut_coulsq + 2.0*rsq_d - 3.0*cut_coul_innersq) / denom_coul;
            double r2inv_d = 1.0 / rsq_d;
            double rinv_d = sqrt(r2inv_d);
            double val_d = qqrd2e * rinv_d * r2inv_d * factor_coul;
            fpair_h += qtmp_h * __float2half((float)(q(j))) * __double2half(switch1_d) * __float2half(val_d);


            /// total float
            // const float r2inv = 1.0f/rsq;
            // const float rinv = sqrtf(r2inv);
            // float forcecoul, switch1;
            // forcecoul = qqrd2e * qtmp * (float)(q(j)) *rinv;
            // switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
            //           (cut_coulsq + 2.0f*rsq - 3.0f*cut_coul_innersq) / denom_coul;
            // forcecoul *= switch1;
            // fpair += forcecoul * r2inv * factor_coul;
            // //fpair_h += __float2half(forcecoul * r2inv * factor_coul);
          }
          else {
            const float r2inv = 1.0f/rsq;
            const float rinv = sqrtf(r2inv);
            float forcecoul, switch1;
            forcecoul = qqrd2e * qtmp * (float)(q(j)) *rinv;
            fpair += forcecoul * r2inv * factor_coul;
          }

        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        fxtmp_h += delx_h*fpair_h;
        fytmp_h += dely_h*fpair_h;
        fztmp_h += delz_h*fpair_h;

        atomicAdd(&f_ptr[j * 3 + 0], (double)( -delx*fpair + __half2float(-delx_h*fpair_h) ));
        atomicAdd(&f_ptr[j * 3 + 1], (double)( -dely*fpair + __half2float(-dely_h*fpair_h) ));
        atomicAdd(&f_ptr[j * 3 + 2], (double)( -delz*fpair + __half2float(-delz_h*fpair_h) ));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < (float)(d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              float englj, switch1;

              englj = r6inv *
                (((float)params(itype,jtype).lj3)*r6inv -
                ((float)params(itype,jtype).lj4));

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < (float)(d_cut_coulsq(itype,jtype))) {
              const float r2inv = 1.0f/rsq;
              const float rinv = sqrtf(r2inv);
              float switch1;

              ecoul = qqrd2e * qtmp * ((float)q(j)) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0f*rsq - 3.0f*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0f*ecoul;
            }
          }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    atomicAdd(&f_ptr[i * 3 + 0], (double)(fxtmp + __half2float(fxtmp_h)));
    atomicAdd(&f_ptr[i * 3 + 1], (double)(fytmp + __half2float(fytmp_h)));
    atomicAdd(&f_ptr[i * 3 + 2], (double)(fztmp + __half2float(fztmp_h)));
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

  PairComputeFunctorCustomFhmix(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomFhmix() {c.copymode = 1; list.copymode = 1;};

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
      fhmix_force_kernel_x_rel_split_cut_inner<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.x_float_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e_h, c.cut_ljsq_h, c.cut_lj_innersq_h, c.denom_lj_h, c.cut_coulsq_h, c.cut_coul_innersq_h, c.denom_coul_h,
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
      // float_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
      //     c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
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
      fhmix_force_kernel_x_rel_split_cut_inner<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel, c.x_float_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e_h, c.cut_ljsq_h, c.cut_lj_innersq_h, c.denom_lj_h, c.cut_coulsq_h, c.cut_coul_innersq_h, c.denom_coul_h,
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
      // float_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
      //     c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
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

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void dfmix_force_kernel_x_rel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float_rel,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x_rel, X_FLOAT binsizex, X_FLOAT binsizey, X_FLOAT binsizez, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e_f, float cut_ljsq_f, float cut_lj_innersq_f, float denom_lj_f, float cut_coulsq_f, float cut_coul_innersq_f, float denom_coul_f,
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double cut_coulsq, double cut_coul_innersq, double denom_coul) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x_rel(i,0);
    const X_FLOAT ytmp = x_rel(i,1);
    const X_FLOAT ztmp = x_rel(i,2);
    const int itype = type(i);
    const F_FLOAT qtmp = q(i);

    const float xtmp_f = x_float_rel(i,0);
    const float ytmp_f = x_float_rel(i,1);
    const float ztmp_f = x_float_rel(i,2);
    const float qtmp_f = q(i);

    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    float fxtmp_f = 0.0f;
    float fytmp_f = 0.0f;
    float fztmp_f = 0.0f;

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

      const float factor_lj_f = 1.0f;
      const float factor_coul_f = 1.0f;
      const float delx_f = xtmp_f - x_float_rel(j,0) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * ((float)binsizex);
      const float dely_f = ytmp_f - x_float_rel(j,1) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * ((float)binsizey);
      const float delz_f = ztmp_f - x_float_rel(j,2) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * ((float)binsizez);
      const float rsq_f = delx_f*delx_f + dely_f*dely_f + delz_f*delz_f;

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();
        float fpair_f = 0.0f;

        if (rsq < d_cut_ljsq(itype,jtype)) {
          //fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          if (rsq > cut_lj_innersq) {
            /// total float 
            const float r2inv_f = 1.0f/rsq_f;
            const float r6inv_f = r2inv_f*r2inv_f*r2inv_f;
            float forcelj_f, switch1_f, switch2_f, englj_f;

            forcelj_f = r6inv_f *
              (((float)params(itype,jtype).lj1)*r6inv_f -
              ((float)params(itype,jtype).lj2));

            switch1_f = (cut_ljsq_f - rsq_f) * (cut_ljsq_f - rsq_f) *
                      (cut_ljsq_f + 2.0f * rsq_f - 3.0f * cut_lj_innersq_f) / denom_lj_f;
            switch2_f = 12.0f * rsq_f * (cut_ljsq_f - rsq_f) * (rsq_f - cut_lj_innersq_f) / denom_lj_f;
            englj_f = r6inv_f *
                    ((float)(params(itype,jtype).lj3)*r6inv_f -
                    (float)(params(itype,jtype).lj4));
            forcelj_f = forcelj_f * switch1_f + englj_f * switch2_f;
            fpair_f += factor_lj_f * forcelj_f * r2inv_f;


            /// total double
            // const F_FLOAT r2inv = 1.0/rsq;
            // const F_FLOAT r6inv = r2inv*r2inv*r2inv;
            // F_FLOAT forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (params(itype,jtype).lj1*r6inv -
            //   params(itype,jtype).lj2);

            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv *
            //         (params(itype,jtype).lj3*r6inv -
            //         params(itype,jtype).lj4);
            // forcelj = forcelj*switch1 + englj*switch2;
            // fpair+=factor_lj*forcelj*r2inv;

          }
          else {
            const F_FLOAT r2inv = 1.0/rsq;
            const F_FLOAT r6inv = r2inv*r2inv*r2inv;
            F_FLOAT forcelj, switch1, switch2, englj;

            forcelj = r6inv *
              (params(itype,jtype).lj1*r6inv -
              params(itype,jtype).lj2);

            fpair+=factor_lj*forcelj*r2inv;
          }
        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
          // fpair+=c.template compute_fcoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
          if (rsq > cut_coul_innersq) {
            /// total float
            const float r2inv_f = 1.0/rsq_f;
            const float rinv_f = sqrtf(r2inv_f);
            float forcecoul_f, switch1_f;

            forcecoul_f = qqrd2e_f * qtmp_f * ((float)q(j)) *rinv_f;
            switch1_f = (cut_coulsq_f-rsq_f) * (cut_coulsq_f-rsq_f) *
                      (cut_coulsq_f + 2.0f*rsq_f - 3.0f*cut_coul_innersq_f) / denom_coul_f;
            forcecoul_f *= switch1_f;
            
            fpair_f += forcecoul_f * r2inv_f * factor_coul_f;

            
            /// total double
            // const F_FLOAT r2inv = 1.0/rsq;
            // const F_FLOAT rinv = sqrt(r2inv);
            // F_FLOAT forcecoul, switch1;

            // forcecoul = qqrd2e * qtmp * q(j) *rinv;
            // switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
            //           (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) / denom_coul;
            // forcecoul *= switch1;
            
            // fpair += forcecoul * r2inv * factor_coul;
          }
          else {
            const F_FLOAT r2inv = 1.0/rsq;
            const F_FLOAT rinv = sqrt(r2inv);
            F_FLOAT forcecoul, switch1;
            
            forcecoul = qqrd2e * qtmp * q(j) *rinv;
            
            fpair += forcecoul * r2inv * factor_coul;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        fxtmp_f += delx_f*fpair_f;
        fytmp_f += dely_f*fpair_f;
        fztmp_f += delz_f*fpair_f;

        // atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        // atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        // atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));
        atomicAdd(&f_ptr[j * 3 + 0], -delx*fpair + (double)(-delx_f*fpair_f));
        atomicAdd(&f_ptr[j * 3 + 1], -dely*fpair + (double)(-dely_f*fpair_f));
        atomicAdd(&f_ptr[j * 3 + 2], -delz*fpair + (double)(-delz_f*fpair_f));

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              F_FLOAT englj, switch1;

              englj = r6inv *
                (params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj4);

              if (rsq > cut_lj_innersq) {
                switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                  (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
                englj *= switch1;
              }

              evdwl = factor_lj * englj;
              ev.evdwl += 1.0*evdwl;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT rinv = sqrt(r2inv);
              F_FLOAT switch1;

              ecoul = qqrd2e * qtmp * q(j) * rinv;
              if (rsq > cut_coul_innersq) {
                switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
                          (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) /
                          denom_coul;
                ecoul *= switch1;
              }

              ecoul *= factor_coul;

              ev.ecoul += 1.0*ecoul;
            }
          }

          // if (vflag_either) {
          //   const E_FLOAT v0 = delx*delx*fpair;
          //   const E_FLOAT v1 = dely*dely*fpair;
          //   const E_FLOAT v2 = delz*delz*fpair;
          //   const E_FLOAT v3 = delx*dely*fpair;
          //   const E_FLOAT v4 = delx*delz*fpair;
          //   const E_FLOAT v5 = dely*delz*fpair;

          //   if (vflag_global) {
          //       ev.v[0] += v0;
          //       ev.v[1] += v1;
          //       ev.v[2] += v2;
          //       ev.v[3] += v3;
          //       ev.v[4] += v4;
          //       ev.v[5] += v5;
          //   }
          // }
        }
      }
    }
    if (EVFLAG) {
      ev_array(ii) = ev;
    }
    // atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    // atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    // atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
    atomicAdd(&f_ptr[i * 3 + 0], fxtmp + (double)(fxtmp_f));
    atomicAdd(&f_ptr[i * 3 + 1], fytmp + (double)(fytmp_f));
    atomicAdd(&f_ptr[i * 3 + 2], fztmp + (double)(fztmp_f));
  }
}

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomDfmix  {
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

  PairComputeFunctorCustomDfmix(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomDfmix() {c.copymode = 1; list.copymode = 1;};

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

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated) {
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
      typename ArrayTypes<device_type>::t_x_array curr_x_rel = c.c_x_rel;
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float_rel = c.x_float_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_rel(i,0) = curr_x(i,0) - curr_bin_base(i,0);
        curr_x_rel(i,1) = curr_x(i,1) - curr_bin_base(i,1);
        curr_x_rel(i,2) = curr_x(i,2) - curr_bin_base(i,2);
        curr_x_float_rel(i,0) = (float)(curr_x(i,0) - curr_bin_base(i,0));
        curr_x_float_rel(i,1) = (float)(curr_x(i,1) - curr_bin_base(i,1));
        curr_x_float_rel(i,2) = (float)(curr_x(i,2) - curr_bin_base(i,2));
      });
      Kokkos::fence();
    }

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      // float_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
      //     c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
      dfmix_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul,
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
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
      typename ArrayTypes<device_type>::t_x_array curr_x_rel = c.c_x_rel;
      Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float_rel = c.x_float_rel;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_bin_base = c.bin_base;
      typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;

      Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
        curr_x_rel(i,0) = curr_x(i,0) - curr_bin_base(i,0);
        curr_x_rel(i,1) = curr_x(i,1) - curr_bin_base(i,1);
        curr_x_rel(i,2) = curr_x(i,2) - curr_bin_base(i,2);
        curr_x_float_rel(i,0) = (float)(curr_x(i,0) - curr_bin_base(i,0));
        curr_x_float_rel(i,1) = (float)(curr_x(i,1) - curr_bin_base(i,1));
        curr_x_float_rel(i,2) = (float)(curr_x(i,2) - curr_bin_base(i,2));
      });
      Kokkos::fence();
    }

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (USE_RELATIVE_COORD) {
      // float_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
      //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
      //     c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
      //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
      //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul);
      dfmix_force_kernel_x_rel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float_rel, c.x_rel, c.binsizex, c.binsizey, c.binsizez,
          c.type, c.q, f, c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.cut_coulsq, (float)c.cut_coul_innersq, (float)c.denom_coul,
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.cut_coulsq, c.cut_coul_innersq, c.denom_coul);
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

  if(NEIGHFLAG != HALFTHREAD) {
    printf("ERROR: NEIGHFLAG is not HALFTHREAD\n");
    exit(1);
  }
  if(ZEROFLAG != 0) {
    printf("ERROR: ZEROFLAG is not 0\n");
    exit(1);
  }
  // if(!std::is_same<Specialisation, void>::value) {
  //   printf("ERROR: Specialisation is not void\n");
  //   exit(1);
  // }

  EV_FLOAT ev;
  if (!fpair->lmp->kokkos->neigh_thread_set)
    if (list->inum <= 16384 && NEIGHFLAG == FULL)
      fpair->lmp->kokkos->neigh_thread = 1;

  if(fpair->lmp->kokkos->neigh_thread != 0) {
    printf("ERROR: NEIGH_THREAD is not zero\n");
    exit(1);
  }

  if (fpair->atom->ntypes < MAX_TYPES_STACKPARAMS) {
    printf("ERROR: atom->ntypes is lesser than MAX_TYPES_STACKPARAMS\n");
    exit(1);
  }

  if (!std::is_same<typename DoCoul<PairStyle::COUL_FLAG>::type, CoulTag>::value) {
    printf("ERROR: DoCoul<PairStyle::COUL_FLAG>::type is not CoulTag\n");
    exit(1);
  }

  if(PRECTYPE == DOUBLE_PREC) {
    //printf("in rhodo double kernel\n");
    PairComputeFunctorCustomDouble<PairStyle,NEIGHFLAG,false, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    //PairComputeFunctor<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list);
    // printf("eflag : %d, vflag : %d, vflag_either : %d, vflag_global : %d\n", fpair->eflag, fpair->vflag, fpair->vflag_either, fpair->vflag_global);
    //if (fpair->eflag || fpair->vflag) {
    if (fpair->eflag) {
      //Kokkos::parallel_reduce(list->inum,ff,ev);
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      //Kokkos::parallel_for(list->inum,ff);
      //ff.test_kernel_launch(list->inum);
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
    //ff.contribute_custom();
  }
  else if(PRECTYPE == FLOAT_PREC) {
    //printf("in rhodo float kernel\n");
    PairComputeFunctorCustomFloat<PairStyle,NEIGHFLAG,false, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
  }
  else if(PRECTYPE == HALF_PREC) {
    printf("in rhodo half kernel\n");
    PairComputeFunctorCustomHalf<PairStyle,NEIGHFLAG,false, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
  }
  else if(PRECTYPE == HFMIX_PREC) {
    //printf("in rhodo fhmix kernel\n");
    PairComputeFunctorCustomFhmix<PairStyle,NEIGHFLAG,false, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
  }
  else if(PRECTYPE == DFMIX_PREC) {
    //printf("in rhodo dfmix kernel\n");
    PairComputeFunctorCustomDfmix<PairStyle,NEIGHFLAG,false, USE_RELATIVE_COORD,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
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
  if(fpair->use_relative_coord) {
    ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, HALFTHREAD, 1, 0, Specialisation> (fpair,list);
  }
  else {
    ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, HALFTHREAD, 0, 0, Specialisation> (fpair,list);
  }
  return ev;
}

} // namespace RhodoKernels
