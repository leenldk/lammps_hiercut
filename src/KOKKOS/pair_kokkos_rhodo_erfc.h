namespace RhodoErfcKernels {

const double EWALD_F =   1.12837917;
const double EWALD_P =   0.3275911;
const double A1      =   0.254829592;
const double A2      =  -0.284496736;
const double A3      =   1.421413741;
const double A4      =  -1.453152027;
const double A5      =   1.061405429;

const float EWALD_F_f =  1.12837917f;
const float EWALD_P_f =  0.3275911f;
const float A1_f      =  0.254829592f;
const float A2_f      = -0.284496736f;
const float A3_f      =  1.421413741f;
const float A4_f      = -1.453152027f;
const float A5_f      =  1.061405429f;

typedef union {
  int i;
  float f;
} union_int_float_t;

struct SpecialVal {
  double coul[4];
  double lj[4];
  SpecialVal(double special_coul[4], double special_lj[4]) {
    for(int i = 0; i < 4; i++) {
      coul[i] = special_coul[i];
      lj[i] = special_lj[i];
    }
  }
};

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void double_force_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  KKScatterView<F_FLOAT*[3], typename ArrayTypes<DeviceType>::t_f_array::array_layout,typename KKDevice<DeviceType>::value,KKScatterSum,typename NeedDup<NEIGHFLAG,DeviceType>::value> dup_f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double tabinnersq,
  int ncoulmask, int ncoulshiftbits, double g_ewald,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

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
      // const F_FLOAT factor_lj = special_lj[j >> SBBITS & 3];
      // const F_FLOAT factor_coul = special_coul[j >> SBBITS & 3];
      const F_FLOAT factor_lj = special.lj[j >> SBBITS & 3];
      const F_FLOAT factor_coul = special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      //const F_FLOAT factor_lj = 1.0;
      //const F_FLOAT factor_coul = 1.0;
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
          // const F_FLOAT r2inv = 1.0/rsq;
          // const F_FLOAT rinv = sqrt(r2inv);
          // F_FLOAT forcecoul, switch1;

          // forcecoul = qqrd2e * qtmp * q(j) *rinv;

          // if (rsq > cut_coul_innersq) {
          //   switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
          //             (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) / denom_coul;
          //   forcecoul *= switch1;
          // }

          // fpair += forcecoul * r2inv * factor_coul;
          
            if (rsq > tabinnersq) {
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
              fpair += forcecoul/rsq;
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

              fpair += forcecoul*rinv*rinv;
            }
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
              // const F_FLOAT r2inv = 1.0/rsq;
              // const F_FLOAT rinv = sqrt(r2inv);
              // F_FLOAT switch1;

              // ecoul = qqrd2e * qtmp * q(j) * rinv;
              // if (rsq > cut_coul_innersq) {
              //   switch1 = (cut_coulsq-rsq) * (cut_coulsq-rsq) *
              //             (cut_coulsq + 2.0*rsq - 3.0*cut_coul_innersq) /
              //             denom_coul;
              //   ecoul *= switch1;

              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
                const F_FLOAT table = d_etable[itable] + fraction*d_detable[itable];
                ecoul = qtmp*q[j] * table;
                if (factor_coul < 1.0) {
                  const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
                  const F_FLOAT prefactor = qtmp*q[j] * table;
                  ecoul -= (1.0-factor_coul)*prefactor;
                }
              } else {
                const F_FLOAT r = sqrt(rsq);
                const F_FLOAT grij = g_ewald * r;
                const F_FLOAT expm2 = exp(-grij*grij);
                const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
                const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const F_FLOAT prefactor = qqrd2e * qtmp*q[j]/r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
              }
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

#define Q_FACTOR 1000

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void double_force_kernel_expr_performance_double(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_doubleq* x_doubleq, typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
  double cutsq, double cut_coulsq,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double tabinnersq,
  int ncoulmask, int ncoulshiftbits, double g_ewald, 
  int ntypes, double2 *lj_param_table, double2 *lj_param_table_upper,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  __shared__ double special_lj_shared[4];
  __shared__ double special_coul_shared[4];
  //__shared__ float2 lj_param_table_upper_shared[TYPE_DIM * (TYPE_DIM + 1) / 2];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = special.lj[i];
      special_coul_shared[i] = special.coul[i];
    }
  }

  // int param_bnd = ntypes * (ntypes + 1) / 2;
  // for(int i = threadIdx.x; i < param_bnd; i += blockDim.x) {
  //   lj_param_table_upper_shared[i] = lj_param_table_upper_f[i];
  // }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    AoS_doubleq x_data_i = x_doubleq[i];
    const X_FLOAT xtmp = x_data_i.x[0];
    const X_FLOAT ytmp = x_data_i.x[1];
    const X_FLOAT ztmp = x_data_i.x[2];
    const int itype = x_data_i.type;
    //const float qtmp = __half2float(x_data_i.q);
    const double qtmp = ((double)x_data_i.q) / Q_FACTOR;
    //const float qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    // if(i == 100) {
    //   const int neigh_stride = &d_neighbors(i,1)-&d_neighbors(i,0);
    //   printf("neigh stride : %d\n", neigh_stride);
    // }

    F_FLOAT fxtmp = 0.0f;
    F_FLOAT fytmp = 0.0f;
    F_FLOAT fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      // const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      // const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      const F_FLOAT factor_lj = special_lj_shared[j >> SBBITS & 3];
      const F_FLOAT factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_doubleq x_data_j = x_doubleq[j];
      const X_FLOAT delx = xtmp - x_data_j.x[0];
      const X_FLOAT dely = ytmp - x_data_j.x[1];
      const X_FLOAT delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      //const float qj = __half2float(x_data_j.q);
      const double qj = ((double)x_data_j.q) / Q_FACTOR;
      //const float qj = q(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      // if (rsq < ((float)d_cutsq(itype,jtype))) {
      if (rsq < cutsq) {

        float fpair = 0.0f;

        //if (rsq < (float)d_cut_ljsq(itype,jtype)) {
        if (rsq < cut_ljsq) {

          const F_FLOAT r2inv = 1.0f/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1, switch2, englj;
          double2 lj_param = lj_param_table_upper[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4

          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            englj = r6inv * (lj_param.x * r6inv - lj_param.y);
            forcelj = forcelj*switch1 + englj*switch2;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
        if (rsq < cut_coulsq) {
            if (rsq > tabinnersq) {

              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
              const F_FLOAT table = d_ftable[itable] + fraction* d_dftable[itable];
              //F_FLOAT forcecoul = qtmp* q[j] * table;
              F_FLOAT forcecoul = qtmp* qj * table;
              if (factor_coul < 1.0) {
                const F_FLOAT table = d_ctable[itable] + fraction* d_dctable[itable];
                //const F_FLOAT prefactor = qtmp* q[j] * table;
                const F_FLOAT prefactor = qtmp* qj * table;
                forcecoul -= (1.0-factor_coul)*prefactor;
              }
              fpair += forcecoul/rsq;
            } else {
              const F_FLOAT r = sqrtf(rsq);
              const F_FLOAT grij = g_ewald * r;
              const F_FLOAT expm2 = expf(-grij*grij);
              const F_FLOAT t = 1.0 / (1.0 + EWALD_P * grij);
              const F_FLOAT rinv = 1.0/r;
              const F_FLOAT erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
              const F_FLOAT prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F * grij * expm2);
              if (factor_coul < 1.0) forcecoul -= (1.0 - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_ptr[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_ptr[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (eflag) {
            // if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
            if (rsq < cut_ljsq) {
              const F_FLOAT r2inv = 1.0 / rsq;
              const F_FLOAT r6inv = r2inv * r2inv * r2inv;
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
            // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq < cut_coulsq) {
              if (rsq > tabinnersq) {

                /// float with table, half in float out
                // ushort itable = __half_as_ushort(__float2half(rsq));
                // float2 etable = coul_etable_f[itable];
                // ecoul = etable.x;
                // if (factor_coul < 1.0) {
                //   ecoul -= (1.0f-factor_coul) * etable.y;
                // }
                // ecoul *= qtmp * qj; 


                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
                const float table = d_etable[itable] + fraction * d_detable[itable];
                ecoul = qtmp* q[j] * table;
                if (factor_coul < 1.0) {
                  const F_FLOAT table = d_ctable[itable] + fraction * d_dctable[itable];
                  const F_FLOAT prefactor = qtmp * q[j] * table;
                  ecoul -= (1.0-factor_coul)*prefactor;
                }
              } else {
                const F_FLOAT r = sqrtf(rsq);
                const F_FLOAT grij = g_ewald * r;
                const F_FLOAT expm2 = expf(-grij*grij);
                const F_FLOAT t = 1.0 / (1.0 + EWALD_P * grij);
                const F_FLOAT erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
                const F_FLOAT prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0) ecoul -= (1.0 - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0 * ecoul;
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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], fztmp);
  }
}

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int ZEROFLAG = 0, class Specialisation = void>
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
  }

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
      const F_FLOAT factor_lj = c.special_lj[sbmask(j)];
      const F_FLOAT factor_coul = c.special_coul[sbmask(j)];
      j &= NEIGHMASK;
      // const F_FLOAT factor_lj = 1.0;
      // const F_FLOAT factor_coul = 1.0;
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

          if (rsq > c.tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & c.ncoulmask) >> c.ncoulshiftbits;
            const F_FLOAT fraction = (rsq_lookup.f - c.d_rtable[itable]) * c.d_drtable[itable];
            const F_FLOAT table = c.d_ftable[itable] + fraction*c.d_dftable[itable];
            F_FLOAT forcecoul = qtmp*c.q[j] * table;
            if (factor_coul < 1.0) {
              const F_FLOAT table = c.d_ctable[itable] + fraction*c.d_dctable[itable];
              const F_FLOAT prefactor = qtmp*c.q[j] * table;
              forcecoul -= (1.0-factor_coul)*prefactor;
            }
            fpair += forcecoul/rsq;
          } else {
            const F_FLOAT r = sqrt(rsq);
            const F_FLOAT grij = c.g_ewald * r;
            const F_FLOAT expm2 = exp(-grij*grij);
            const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
            const F_FLOAT rinv = 1.0/r;
            const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            const F_FLOAT prefactor = c.qqrd2e * qtmp*c.q[j]*rinv;
            F_FLOAT forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
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
              //ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);

              if (rsq > c.tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & c.ncoulmask) >> c.ncoulshiftbits;
                const F_FLOAT fraction = (rsq_lookup.f - c.d_rtable[itable]) * c.d_drtable[itable];
                const F_FLOAT table = c.d_etable[itable] + fraction*c.d_detable[itable];
                ecoul = qtmp*c.q[j] * table;
                if (factor_coul < 1.0) {
                  const F_FLOAT table = c.d_ctable[itable] + fraction*c.d_dctable[itable];
                  const F_FLOAT prefactor = qtmp*c.q[j] * table;
                  ecoul -= (1.0-factor_coul)*prefactor;
                }
              } else {
                const F_FLOAT r = sqrt(rsq);
                const F_FLOAT grij = c.g_ewald * r;
                const F_FLOAT expm2 = exp(-grij*grij);
                const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
                const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const F_FLOAT prefactor = c.qqrd2e * qtmp*c.q[j]/r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
              }
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
    // //else compute_item_custom<0,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    //compute_item<0,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &energy_virial) const {
    // //if (c.newton_pair)
    energy_virial += compute_item_custom<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    // //else
    // //  energy_virial += compute_item_custom<1,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    //energy_virial += compute_item<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }
  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }
    if(fpair -> x_doubleq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_doubleq\n"); fflush(stdout);
      if(fpair -> x_doubleq_size > 0) {
        cudaFree(fpair -> x_doubleq);
      }
      cudaMalloc((void**)&(fpair -> x_doubleq), (fpair -> x).extent(0) * sizeof(AoS_doubleq));
      fpair -> x_doubleq_size = (fpair -> x).extent(0);
      c.x_doubleq = fpair -> x_doubleq;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;
    typename ArrayTypes<device_type>::t_int_1d_randomread curr_type = c.type;
    typename ArrayTypes<device_type>::t_float_1d_randomread curr_q = c.q;
    AoS_doubleq* curr_x_doubleq = c.x_doubleq;

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      AoS_doubleq temp;
      temp.x[0] = curr_x(i,0);
      temp.x[1] = curr_x(i,1);
      temp.x[2] = curr_x(i,2);
      temp.type = curr_type(i);
      temp.q = curr_q(i) * Q_FACTOR;
      curr_x_doubleq[i] = temp;
    });
    Kokkos::fence();

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    // double_force_kernel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, c.q, dup_f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, c.g_ewald,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    double_force_kernel_expr_performance_double<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
        ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_doubleq, c.q, f,
        SpecialVal(c.special_coul, c.special_lj),
        c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.params, 
        c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
        c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
        c.ncoulmask, c.ncoulshiftbits, c.g_ewald, 
        (c.atom)->ntypes, c.lj_param_table, c.lj_param_table_upper,
        c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    cudaDeviceSynchronize();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }
    if(fpair -> x_doubleq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_doubleq\n"); fflush(stdout);
      if(fpair -> x_doubleq_size > 0) {
        cudaFree(fpair -> x_doubleq);
      }
      cudaMalloc((void**)&(fpair -> x_doubleq), (fpair -> x).extent(0) * sizeof(AoS_doubleq));
      fpair -> x_doubleq_size = (fpair -> x).extent(0);
      c.x_doubleq = fpair -> x_doubleq;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;
    typename ArrayTypes<device_type>::t_int_1d_randomread curr_type = c.type;
    typename ArrayTypes<device_type>::t_float_1d_randomread curr_q = c.q;
    AoS_doubleq* curr_x_doubleq = c.x_doubleq;

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      AoS_doubleq temp;
      temp.x[0] = curr_x(i,0);
      temp.x[1] = curr_x(i,1);
      temp.x[2] = curr_x(i,2);
      temp.type = curr_type(i);
      temp.q = curr_q(i) * Q_FACTOR;
      curr_x_doubleq[i] = temp;
    });
    Kokkos::fence();

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    // double_force_kernel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, c.q, dup_f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, c.g_ewald,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    double_force_kernel_expr_performance_double<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
        ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_doubleq, c.q, f,
        SpecialVal(c.special_coul, c.special_lj),
        c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.params, 
        c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
        c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
        c.ncoulmask, c.ncoulshiftbits, c.g_ewald, 
        (c.atom)->ntypes, c.lj_param_table, c.lj_param_table_upper,
        c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

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
__global__ void float_force_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = (float)q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (((float)params(itype,jtype).lj1)*r6inv -
            ((float)params(itype,jtype).lj2));

          if (rsq > ((float)cut_lj_innersq)) {
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
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq > tabinnersq) {
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              float forcecoul = qtmp* ((float)q[j]) * table;
              if (factor_coul < 1.0) {
                const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
                const float prefactor = qtmp* ((float)q[j]) * table;
                forcecoul -= (1.0f-factor_coul)*prefactor;
              }
              fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * ((float)q[j]) * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                ecoul = qtmp* ((float)q[j]) * table;
                if (factor_coul < 1.0f) {
                  const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                  const float prefactor = qtmp * ((float)q[j]) * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * ((float)q[j]) / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_sim_half_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = (float)q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          if (rsq > ((float)cut_lj_innersq)) {
            /// float with table, half in half out
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, englj;

            // const __half rsq_h = __float2half(rsq);
            // double rsq_d = (double)__half2float(rsq_h);
            // double switch1_d = (cut_ljsq-rsq_d) * (cut_ljsq-rsq_d) *
            //           (cut_ljsq + 2.0*rsq_d - 3.0*cut_lj_innersq) / denom_lj;
            // double switch2_d = 12.0*rsq_d * (cut_ljsq-rsq_d) * (rsq_d-cut_lj_innersq) / denom_lj;
            // double r2inv_d = 1.0 / rsq_d;
            // double r6inv_d = r2inv_d*r2inv_d*r2inv_d;
            // __half f1_rsq = __double2half(r6inv_d * switch1_d * r2inv_d);
            // __half f2_rsq = __double2half(r6inv_d * switch2_d * r2inv_d);
            // forcelj = __half2float(f1_rsq) * (((float)params(itype,jtype).lj1)*r6inv - ((float)params(itype,jtype).lj2)) + 
            //           __half2float(f2_rsq) * (((float)params(itype,jtype).lj3)*r6inv - ((float)params(itype,jtype).lj4));

            // // float f1_rsq = r6inv_d * switch1_d * r2inv_d;
            // // float f2_rsq = r6inv_d * switch2_d * r2inv_d;
            // // forcelj = f1_rsq * (((float)params(itype,jtype).lj1)*r6inv - ((float)params(itype,jtype).lj2)) + 
            // //           f2_rsq * (((float)params(itype,jtype).lj3)*r6inv - ((float)params(itype,jtype).lj4));

            // fpair+=factor_lj*forcelj;


            /// float with table, half in float out 
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, englj;

            // const __half rsq_h = __float2half(rsq);
            // double rsq_d = (double)__half2float(rsq_h);
            // double switch1_d = (cut_ljsq-rsq_d) * (cut_ljsq-rsq_d) *
            //           (cut_ljsq + 2.0*rsq_d - 3.0*cut_lj_innersq) / denom_lj;
            // double switch2_d = 12.0*rsq_d * (cut_ljsq-rsq_d) * (rsq_d-cut_lj_innersq) / denom_lj;
            // double r2inv_d = 1.0 / rsq_d;
            // double r6inv_d = r2inv_d*r2inv_d*r2inv_d;
            // double f1_rsq = r6inv_d * switch1_d * r2inv_d;
            // double f2_rsq = r6inv_d * switch2_d * r2inv_d;

            // forcelj = f1_rsq * (((float)params(itype,jtype).lj1)*r6inv - ((float)params(itype,jtype).lj2)) + 
            //           f2_rsq * (((float)params(itype,jtype).lj3)*r6inv - ((float)params(itype,jtype).lj4));
            // fpair+=factor_lj*forcelj;


            /// float with half lj
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;
            // __half lj1_h = __double2half(params(itype,jtype).lj1);
            // __half lj2_h = __double2half(params(itype,jtype).lj2);
            // __half lj3_h = __double2half(params(itype,jtype).lj3);
            // __half lj4_h = __double2half(params(itype,jtype).lj4);

            // forcelj = r6inv *
            //   (__half2float(lj1_h) * r6inv -
            //   __half2float(lj2_h));

            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv *
            //         (__half2float(lj3_h) * r6inv -
            //         __half2float(lj4_h));
            // forcelj = forcelj*switch1 + englj*switch2;

            // fpair+=factor_lj*forcelj*r2inv;


            /// float with half lj, only lj1, lj3
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;
            // __half lj1_h = __double2half(params(itype,jtype).lj1);
            // float lj2_f = (float)(params(itype,jtype).lj2);
            // __half lj3_h = __double2half(params(itype,jtype).lj3);
            // float lj4_f = (float)(params(itype,jtype).lj4);

            // forcelj = r6inv *
            //   (__half2float(lj1_h) * r6inv -
            //   lj2_f);

            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv *
            //         (__half2float(lj3_h) * r6inv -
            //         lj4_f);
            // forcelj = forcelj*switch1 + englj*switch2;

            // fpair+=factor_lj*forcelj*r2inv;


            /// total float, calc lj1 and lj2
            const float r2inv = 1.0f/rsq;
            const float r6inv = r2inv*r2inv*r2inv;
            float forcelj, switch1, switch2, englj;
            float lj3_f = (float)(params(itype,jtype).lj3);
            float lj4_f = (float)(params(itype,jtype).lj4);
            float lj1_f = lj3_f * 12.0;
            float lj2_f = lj4_f * 6.0;

            forcelj = r6inv *
              (lj1_f * r6inv -
              lj2_f);

            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            englj = r6inv *
                    (lj3_f * r6inv -
                    lj4_f);
            forcelj = forcelj*switch1 + englj*switch2;

            fpair+=factor_lj*forcelj*r2inv;


            /// total float
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv *
            //         (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj*switch1 + englj*switch2;

            // fpair+=factor_lj*forcelj*r2inv;
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
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq > tabinnersq) {
              /// float with table, half in float out
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // __half rsq_h = __float2half(rsq);
              // double rsq_d = (double)__half2float(rsq_h);
              // const double fraction = (rsq_d - d_rtable[itable]) * d_drtable[itable];
              // const double table = d_ftable[itable] + fraction* d_dftable[itable];
              // const float f1_rsq = table / rsq_d;
              // float forcecoul = f1_rsq;
              // if (factor_coul < 1.0) {
              //   double table1 = d_ctable[itable] + fraction* d_dctable[itable];
              //   const float f2_rsq = table1 / rsq_d;
              //   forcecoul -= (1.0f-factor_coul) * f2_rsq;
              // }
              // fpair += qtmp * ((float)q[j]) * forcecoul;


              /// float with table, half in half out
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              __half rsq_h = __float2half(rsq);
              double rsq_d = (double)__half2float(rsq_h);
              const double fraction = (rsq_d - d_rtable[itable]) * d_drtable[itable];
              const double table = d_ftable[itable] + fraction* d_dftable[itable];
              const __half f1_rsq = __double2half(table / rsq_d);
              float forcecoul = __half2float(f1_rsq);
              if (factor_coul < 1.0) {
                double table1 = d_ctable[itable] + fraction* d_dctable[itable];
                const __half f2_rsq = __double2half(table1 / rsq_d);
                forcecoul -= (1.0f-factor_coul) * __half2float(f2_rsq);
              }
              fpair += qtmp * ((float)q[j]) * forcecoul;


              /// total float 
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              // const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              // float forcecoul = qtmp* ((float)q[j]) * table;
              // if (factor_coul < 1.0) {
              //   const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
              //   const float prefactor = qtmp* ((float)q[j]) * table;
              //   forcecoul -= (1.0f-factor_coul)*prefactor;
              // }
              // fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * ((float)q[j]) * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                ecoul = qtmp* ((float)q[j]) * table;
                if (factor_coul < 1.0f) {
                  const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                  const float prefactor = qtmp * ((float)q[j]) * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * ((float)q[j]) / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_half_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, __half2 *coul_ftable, float2 *coul_etable_f,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = (float)q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;

          forcelj = r6inv *
            (((float)params(itype,jtype).lj1)*r6inv -
            ((float)params(itype,jtype).lj2));

          if (rsq > ((float)cut_lj_innersq)) {
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
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq > tabinnersq) {
              /// float with table, half in half out
              ushort itable = __half_as_ushort(__float2half(rsq));
              __half2 ftable = coul_ftable[itable];
              float forcecoul = __half2float(ftable.x);
              if (factor_coul < 1.0) {
                forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              }
              fpair += qtmp * ((float)q[j]) * forcecoul;

              /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              // const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              // float forcecoul = qtmp* ((float)q[j]) * table;
              // if (factor_coul < 1.0) {
              //   const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
              //   const float prefactor = qtmp* ((float)q[j]) * table;
              //   forcecoul -= (1.0f-factor_coul)*prefactor;
              // }
              // fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * ((float)q[j]) * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              if (rsq > tabinnersq) {
                /// float with table, half in half out
                // ushort itable = __half_as_ushort(__float2half(rsq));
                // __half2 etable = coul_etable[itable];
                // ecoul = __half2float(etable.x);
                // if (factor_coul < 1.0) {
                //   ecoul -= (1.0f-factor_coul) * __half2float(etable.y);
                // }
                // ecoul *= qtmp * ((float)q[j]);                


                /// float with table, half in float out
                ushort itable = __half_as_ushort(__float2half(rsq));
                float2 etable = coul_etable_f[itable];
                ecoul = etable.x;
                if (factor_coul < 1.0) {
                  ecoul -= (1.0f-factor_coul) * etable.y;
                }
                ecoul *= qtmp * ((float)q[j]); 


                /// total float 
                // union_int_float_t rsq_lookup;
                // rsq_lookup.f = rsq;
                // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                // const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                // ecoul = qtmp* ((float)q[j]) * table;
                // if (factor_coul < 1.0f) {
                //   const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                //   const float prefactor = qtmp * ((float)q[j]) * table;
                //   ecoul -= (1.0f-factor_coul)*prefactor;
                // }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * ((float)q[j]) / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_half_table_no_dtable(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  __half2 *coul_ftable, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int lj_param_dim_size, float2 *lj_param_table_f) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = (float)q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          /// float with table, half in float out 
          // const float r2inv = 1.0f/rsq;
          // const float r6inv = r2inv*r2inv*r2inv;
          // float forcelj, switch1, switch2, englj;

          // forcelj = r6inv * (((float)params(itype,jtype).lj1)*r6inv -
          //           ((float)params(itype,jtype).lj2));

          // if (rsq > ((float)cut_lj_innersq)) {
          //   float2 etable = lj_ftable_f[__half_as_ushort(__float2half(rsq))];
          //   englj = (((float)params(itype,jtype).lj3)*r6inv -
          //           ((float)params(itype,jtype).lj4));
          //   forcelj = forcelj * etable.x + englj * etable.y;
          // }

          // fpair+=factor_lj*forcelj*r2inv;


          /// float with table, half in half out 
          // const float r2inv = 1.0f/rsq;
          // const float r6inv = r2inv*r2inv*r2inv;
          // float forcelj, switch1, switch2, englj;

          // forcelj = r6inv * (((float)params(itype,jtype).lj1)*r6inv -
          //           ((float)params(itype,jtype).lj2));

          // if (rsq > ((float)cut_lj_innersq)) {
          //   __half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
          //   englj = (((float)params(itype,jtype).lj3)*r6inv -
          //           ((float)params(itype,jtype).lj4));
          //   forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
          // }

          // fpair+=factor_lj*forcelj*r2inv;


          /// float with table, half in half out, half lj
          // const float r2inv = 1.0f/rsq;
          // const float r6inv = r2inv*r2inv*r2inv;
          // float forcelj, switch1, switch2, englj;
          // __half lj1_h = __double2half(params(itype,jtype).lj1);
          // __half lj2_h = __double2half(params(itype,jtype).lj2);
          // __half lj3_h = __double2half(params(itype,jtype).lj3);
          // __half lj4_h = __double2half(params(itype,jtype).lj4);

          // forcelj = r6inv * ( __half2float(lj1_h) *r6inv -
          //           __half2float(lj2_h));

          // if (rsq > ((float)cut_lj_innersq)) {
          //   __half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
          //   englj = ( __half2float(lj3_h) *r6inv -
          //           __half2float(lj4_h));
          //   forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
          // }

          // fpair+=factor_lj*forcelj*r2inv;


          /// float with table, half in half out, params.lj from table
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;
          float2 lj_param = lj_param_table_f[itype * lj_param_dim_size + jtype];  // lj3, lj4

          // forcelj = r6inv * (((float)params(itype,jtype).lj1)*r6inv -
          //           ((float)params(itype,jtype).lj2));
          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

          if (rsq > ((float)cut_lj_innersq)) {
            __half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
            englj = (lj_param.x * r6inv - lj_param.y);
            forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
          }

          fpair+=factor_lj*forcelj*r2inv;


          /// total float
          // const float r2inv = 1.0f/rsq;
          // const float r6inv = r2inv*r2inv*r2inv;
          // float forcelj, switch1, switch2, englj;

          // forcelj = r6inv *
          //   (((float)params(itype,jtype).lj1)*r6inv -
          //   ((float)params(itype,jtype).lj2));

          // if (rsq > ((float)cut_lj_innersq)) {
          //   switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
          //             (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
          //   switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
          //   englj = r6inv *
          //           (((float)params(itype,jtype).lj3)*r6inv -
          //           ((float)params(itype,jtype).lj4));
          //   forcelj = forcelj*switch1 + englj*switch2;
          // }

          // fpair+=factor_lj*forcelj*r2inv;

        }
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq > tabinnersq) {
              /// float with table, half in half out
              ushort itable = __half_as_ushort(__float2half(rsq));
              __half2 ftable = coul_ftable[itable];
              float forcecoul = __half2float(ftable.x);
              if (factor_coul < 1.0) {
                forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              }
              fpair += qtmp * ((float)q[j]) * forcecoul;

              /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              // const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              // float forcecoul = qtmp* ((float)q[j]) * table;
              // if (factor_coul < 1.0) {
              //   const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
              //   const float prefactor = qtmp* ((float)q[j]) * table;
              //   forcecoul -= (1.0f-factor_coul)*prefactor;
              // }
              // fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * ((float)q[j]) * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              if (rsq > tabinnersq) {
                /// float with table, half in half out
                // ushort itable = __half_as_ushort(__float2half(rsq));
                // __half2 etable = coul_etable[itable];
                // ecoul = __half2float(etable.x);
                // if (factor_coul < 1.0) {
                //   ecoul -= (1.0f-factor_coul) * __half2float(etable.y);
                // }
                // ecoul *= qtmp * ((float)q[j]);                


                /// float with table, half in float out
                ushort itable = __half_as_ushort(__float2half(rsq));
                float2 etable = coul_etable_f[itable];
                ecoul = etable.x;
                if (factor_coul < 1.0) {
                  ecoul -= (1.0f-factor_coul) * etable.y;
                }
                ecoul *= qtmp * ((float)q[j]); 


                /// total float 
                // union_int_float_t rsq_lookup;
                // rsq_lookup.f = rsq;
                // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                // const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                // ecoul = qtmp* ((float)q[j]) * table;
                // if (factor_coul < 1.0f) {
                //   const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                //   const float prefactor = qtmp * ((float)q[j]) * table;
                //   ecoul -= (1.0f-factor_coul)*prefactor;
                // }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * ((float)q[j]) / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}

#define LJ_LBND    0b0101010000000000 // half(64.0)
#define COUL_LBND  0b0100000000000000 // half(2.0)
#define R_BND      0b0101100010000001 // half(144.0 + eps) 
#define TYPE_DIM   68 // atom types
#define LJ_BITS    2
#define LJ_OFFSET  2
#define COUL_BITS  2
#define COUL_OFFSET 2

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_half_table_no_dtable_shared_mem(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  __half2 *coul_ftable, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int lj_param_dim_size, float2 *lj_param_table_f) {

  __shared__ __half2 lj_ftable_shared[R_BND - LJ_LBND];
  __shared__ __half2 coul_ftable_shared[R_BND - COUL_LBND];

  for(ushort i = LJ_LBND + threadIdx.x; i < R_BND; i += blockDim.x) {
    lj_ftable_shared[i - LJ_LBND] = lj_ftable[i];
  }
  for(ushort i = COUL_LBND + threadIdx.x; i < R_BND; i += blockDim.x) {
    coul_ftable_shared[i - COUL_LBND] = coul_ftable[i];
  }
  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = (float)q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          /// float with table, half in half out, params.lj from table
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;
          float2 lj_param = lj_param_table_f[itype * lj_param_dim_size + jtype];  // lj3, lj4

          // forcelj = r6inv * (((float)params(itype,jtype).lj1)*r6inv -
          //           ((float)params(itype,jtype).lj2));
          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

          if (rsq > ((float)cut_lj_innersq)) {
            //__half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
            __half2 etable = lj_ftable_shared[__half_as_ushort(__float2half(rsq)) - LJ_LBND];
            englj = (lj_param.x * r6inv - lj_param.y);
            forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq > tabinnersq) {
              /// float with table, half in half out
              ushort itable = __half_as_ushort(__float2half(rsq));
              //__half2 ftable = coul_ftable[itable];
              __half2 ftable = coul_ftable_shared[itable - COUL_LBND];
              float forcecoul = __half2float(ftable.x);
              if (factor_coul < 1.0) {
                forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              }
              fpair += qtmp * ((float)q[j]) * forcecoul;

              /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              // const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              // float forcecoul = qtmp* ((float)q[j]) * table;
              // if (factor_coul < 1.0) {
              //   const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
              //   const float prefactor = qtmp* ((float)q[j]) * table;
              //   forcecoul -= (1.0f-factor_coul)*prefactor;
              // }
              // fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * ((float)q[j]) * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              if (rsq > tabinnersq) {
                /// float with table, half in half out
                // ushort itable = __half_as_ushort(__float2half(rsq));
                // __half2 etable = coul_etable[itable];
                // ecoul = __half2float(etable.x);
                // if (factor_coul < 1.0) {
                //   ecoul -= (1.0f-factor_coul) * __half2float(etable.y);
                // }
                // ecoul *= qtmp * ((float)q[j]);                


                /// float with table, half in float out
                ushort itable = __half_as_ushort(__float2half(rsq));
                float2 etable = coul_etable_f[itable];
                ecoul = etable.x;
                if (factor_coul < 1.0) {
                  ecoul -= (1.0f-factor_coul) * etable.y;
                }
                ecoul *= qtmp * ((float)q[j]); 


                /// total float 
                // union_int_float_t rsq_lookup;
                // rsq_lookup.f = rsq;
                // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                // const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                // ecoul = qtmp* ((float)q[j]) * table;
                // if (factor_coul < 1.0f) {
                //   const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                //   const float prefactor = qtmp * ((float)q[j]) * table;
                //   ecoul -= (1.0f-factor_coul)*prefactor;
                // }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * ((float)q[j]) / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_half_table_no_dtable_shared_mem_aos(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_floatq* x_floatq, typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
  // typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  // typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  // typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  __half2 *coul_ftable, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f) {

  __shared__ __half2 lj_ftable_shared[R_BND - LJ_LBND];
  __shared__ __half2 coul_ftable_shared[R_BND - COUL_LBND];
  //__shared__ float2 lj_param_table_shared[TYPE_DIM * TYPE_DIM];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];
  __shared__ float2 lj_param_table_upper_shared[TYPE_DIM * (TYPE_DIM + 1) / 2];

  for(ushort i = LJ_LBND + threadIdx.x; i < R_BND; i += blockDim.x) {
    lj_ftable_shared[i - LJ_LBND] = lj_ftable[i];
  }
  for(ushort i = COUL_LBND + threadIdx.x; i < R_BND; i += blockDim.x) {
    coul_ftable_shared[i - COUL_LBND] = coul_ftable[i];
  }
  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  int param_bnd = ntypes * (ntypes + 1) / 2;
  for(int i = threadIdx.x; i < param_bnd; i += blockDim.x) {
    lj_param_table_upper_shared[i] = lj_param_table_upper_f[i];
  }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];
    const int itype = (int)(x_data_i.type);
    //const float qtmp = __half2float(x_data_i.q);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    //const float qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    // if(i == 100) {
    //   const int neigh_stride = &d_neighbors(i,1)-&d_neighbors(i,0);
    //   printf("neigh stride : %d\n", neigh_stride);
    // }

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      // const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      // const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      //const float qj = __half2float(x_data_j.q);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      //const float qj = q(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      // if (rsq < ((float)d_cutsq(itype,jtype))) {
      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        //if (rsq < (float)d_cut_ljsq(itype,jtype)) {
        if (rsq < cut_ljsq_f) {
          /// float with table, half in half out, params.lj from table
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;
          //float2 lj_param = lj_param_table_f[itype * lj_param_dim_size + jtype];  // lj3, lj4
          //float2 lj_param = lj_param_table_shared[min(itype, jtype) * lj_param_dim_size + max(itype, jtype)];  // lj3, lj4
          float2 lj_param = lj_param_table_upper_f[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4
          //float2 lj_param = lj_param_table_upper_shared[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4

          // forcelj = r6inv * (((float)params(itype,jtype).lj1)*r6inv -
          //           ((float)params(itype,jtype).lj2));
          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

          if (rsq > ((float)cut_lj_innersq)) {
            //__half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
            __half2 etable = lj_ftable_shared[__half_as_ushort(__float2half(rsq)) - LJ_LBND];
            englj = (lj_param.x * r6inv - lj_param.y);
            forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
              /// float with table, half in half out
              ushort itable = __half_as_ushort(__float2half(rsq));
              //__half2 ftable = coul_ftable[itable];
              __half2 ftable = coul_ftable_shared[itable - COUL_LBND];
              float forcecoul = __half2float(ftable.x);
              if (factor_coul < 1.0) {
                forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              }
              fpair += qtmp * qj * forcecoul;

              /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              // const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              // float forcecoul = qtmp* ((float)q[j]) * table;
              // if (factor_coul < 1.0) {
              //   const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
              //   const float prefactor = qtmp* ((float)q[j]) * table;
              //   forcecoul -= (1.0f-factor_coul)*prefactor;
              // }
              // fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            // if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
            if (rsq < cut_ljsq_f) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// float with table, half in half out
                // ushort itable = __half_as_ushort(__float2half(rsq));
                // __half2 etable = coul_etable[itable];
                // ecoul = __half2float(etable.x);
                // if (factor_coul < 1.0) {
                //   ecoul -= (1.0f-factor_coul) * __half2float(etable.y);
                // }
                // ecoul *= qtmp * ((float)q[j]);                


                /// float with table, half in float out
                ushort itable = __half_as_ushort(__float2half(rsq));
                float2 etable = coul_etable_f[itable];
                ecoul = etable.x;
                if (factor_coul < 1.0) {
                  ecoul -= (1.0f-factor_coul) * etable.y;
                }
                ecoul *= qtmp * qj; 


                /// total float 
                // union_int_float_t rsq_lookup;
                // rsq_lookup.f = rsq;
                // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                // const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                // ecoul = qtmp* ((float)q[j]) * table;
                // if (factor_coul < 1.0f) {
                //   const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                //   const float prefactor = qtmp * ((float)q[j]) * table;
                //   ecoul -= (1.0f-factor_coul)*prefactor;
                // }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}


template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_expr_half_table_comp(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  __half2 *lj_ftable, float2 *lj_ftable_f, 
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = (float)q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          if (rsq > ((float)cut_lj_innersq)) {

            /// half calc
            const float r2inv = 1.0f/rsq;
            const float r6inv = r2inv*r2inv*r2inv;
            float forcelj, switch1, switch2, englj;
            __half rsq_h = __float2half(rsq);
            __half cut_ljsq_h = __float2half(cut_ljsq);
            __half denom_lj_h = __float2half(denom_lj);
            __half r6inv_h = __float2half(r6inv);
            __half cut_lj_innersq_h = __float2half(cut_lj_innersq);

            forcelj = r6inv *
              (((float)params(itype,jtype).lj1)*r6inv -
              ((float)params(itype,jtype).lj2));

            // float val1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // float val2 = r6inv * 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            __half val1_h = (cut_ljsq_h - rsq_h) * (cut_ljsq_h - rsq_h) *
                      (cut_ljsq_h + __float2half(2.0f) * rsq_h - __float2half(3.0f) * cut_lj_innersq_h) / denom_lj_h;
            __half val2_h = r6inv_h * __float2half(12.0f) * rsq_h * (cut_ljsq_h - rsq_h) * (rsq_h - cut_lj_innersq_h) / denom_lj_h;
            englj = (((float)params(itype,jtype).lj3)*r6inv -
                    ((float)params(itype,jtype).lj4));
            //forcelj = forcelj * val1 + englj * val2;
            forcelj = forcelj * __half2float(val1_h) + englj * __half2float(val2_h);

            fpair+=factor_lj*forcelj*r2inv;


            /// half table
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // __half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            // fpair+=factor_lj*forcelj*r2inv;


            /// total float
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv *
            //         (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj*switch1 + englj*switch2;

            // fpair+=factor_lj*forcelj*r2inv;
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
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq > tabinnersq) {
              /// total float 
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              float forcecoul = qtmp* ((float)q[j]) * table;
              if (factor_coul < 1.0) {
                const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
                const float prefactor = qtmp* ((float)q[j]) * table;
                forcecoul -= (1.0f-factor_coul)*prefactor;
              }
              fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * ((float)q[j]) * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                ecoul = qtmp* ((float)q[j]) * table;
                if (factor_coul < 1.0f) {
                  const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                  const float prefactor = qtmp * ((float)q[j]) * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * ((float)q[j]) / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}


template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_expr_part_comp(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  __half2 *coul_ftable, float2 *coul_ftable_f, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = (float)q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < (float)d_cut_ljsq(itype,jtype)) {
          if (rsq > ((float)cut_lj_innersq)) {

            /// LJ half table empty outer 


            /// LJ half table truncate 8 bit 
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // __half2 etable = lj_ftable[(__half_as_ushort(__float2half(rsq)) + 128) & 0xff00];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            // fpair+=factor_lj*forcelj*r2inv;


            /// LJ half table truncate 6 bit 
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // __half2 etable = lj_ftable[(__half_as_ushort(__float2half(rsq)) + 32) & 0xffc0];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            // fpair+=factor_lj*forcelj*r2inv;


            /// LJ half table truncate 4 bit 
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // __half2 etable = lj_ftable[(__half_as_ushort(__float2half(rsq)) + 8) & 0xfff0];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            // fpair+=factor_lj*forcelj*r2inv;


            /// LJ half table truncate 3 bit 
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // __half2 etable = lj_ftable[(__half_as_ushort(__float2half(rsq)) + 4) & 0xfff8];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            // fpair+=factor_lj*forcelj*r2inv;


            /// LJ half table truncate 2 bit 
            const float r2inv = 1.0f/rsq;
            const float r6inv = r2inv*r2inv*r2inv;
            float forcelj, switch1, switch2, englj;

            forcelj = r6inv *
              (((float)params(itype,jtype).lj1)*r6inv -
              ((float)params(itype,jtype).lj2));

            __half2 etable = lj_ftable[(__half_as_ushort(__float2half(rsq)) + 2) & 0xfffc];
            englj = (((float)params(itype,jtype).lj3)*r6inv -
                    ((float)params(itype,jtype).lj4));
            forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            fpair+=factor_lj*forcelj*r2inv;


            /// LJ half table truncate 1 bit 
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // __half2 etable = lj_ftable[(__half_as_ushort(__float2half(rsq)) + 1) & 0xfffe];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            // fpair+=factor_lj*forcelj*r2inv;


            /// LJ table half in float out
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // float2 etable = lj_ftable_f[__half_as_ushort(__float2half(rsq))];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * etable.x + englj * etable.y;

            // fpair+=factor_lj*forcelj*r2inv;


            /// LJ table half in half out
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // __half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
            // englj = (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            // fpair+=factor_lj*forcelj*r2inv;


            /// total float
            // const float r2inv = 1.0f/rsq;
            // const float r6inv = r2inv*r2inv*r2inv;
            // float forcelj, switch1, switch2, englj;

            // forcelj = r6inv *
            //   (((float)params(itype,jtype).lj1)*r6inv -
            //   ((float)params(itype,jtype).lj2));

            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv *
            //         (((float)params(itype,jtype).lj3)*r6inv -
            //         ((float)params(itype,jtype).lj4));
            // forcelj = forcelj*switch1 + englj*switch2;

            // fpair+=factor_lj*forcelj*r2inv;
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
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq > tabinnersq) {

              /// Coul half table truncate 4 bit 
              // ushort itable = __half_as_ushort(__float2half(rsq));
              // __half2 ftable = coul_ftable[(itable + 8) & 0xfff0];
              // float forcecoul = __half2float(ftable.x);
              // if (factor_coul < 1.0) {
              //   forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              // }
              // fpair += qtmp * ((float)q[j]) * forcecoul;


              /// Coul half table truncate 3 bit 
              // ushort itable = __half_as_ushort(__float2half(rsq));
              // __half2 ftable = coul_ftable[(itable + 4) & 0xfff8];
              // float forcecoul = __half2float(ftable.x);
              // if (factor_coul < 1.0) {
              //   forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              // }
              // fpair += qtmp * ((float)q[j]) * forcecoul;


              /// Coul half table truncate 2 bit 
              ushort itable = __half_as_ushort(__float2half(rsq));
              __half2 ftable = coul_ftable[(itable + 2) & 0xfffc];
              float forcecoul = __half2float(ftable.x);
              if (factor_coul < 1.0) {
                forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              }
              fpair += qtmp * ((float)q[j]) * forcecoul;


              /// Coul table half in half out
              // ushort itable = __half_as_ushort(__float2half(rsq));
              // __half2 ftable = coul_ftable[itable];
              // float forcecoul = __half2float(ftable.x);
              // if (factor_coul < 1.0) {
              //   forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              // }
              // fpair += qtmp * ((float)q[j]) * forcecoul;


              /// Coul table half in float out
              // ushort itable = __half_as_ushort(__float2half(rsq));
              // //__half2 ftable = coul_ftable[itable];
              // float2 ftable = coul_ftable_f[itable];
              // float forcecoul = ftable.x;
              // if (factor_coul < 1.0) {
              //   forcecoul -= (1.0f-factor_coul) * ftable.y;
              // }
              // fpair += qtmp * ((float)q[j]) * forcecoul;

              /// total float 
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              // const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              // float forcecoul = qtmp* ((float)q[j]) * table;
              // if (factor_coul < 1.0) {
              //   const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
              //   const float prefactor = qtmp* ((float)q[j]) * table;
              //   forcecoul -= (1.0f-factor_coul)*prefactor;
              // }
              // fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * ((float)q[j]) * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              if (rsq > tabinnersq) {
                /// Coul etable
                ushort itable = __half_as_ushort(__float2half(rsq));
                float2 etable = coul_etable_f[itable];
                ecoul = etable.x;
                if (factor_coul < 1.0) {
                  ecoul -= (1.0f-factor_coul) * etable.y;
                }
                ecoul *= qtmp * ((float)q[j]); 


                /// total float 
                // union_int_float_t rsq_lookup;
                // rsq_lookup.f = rsq;
                // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                // const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                // ecoul = qtmp* ((float)q[j]) * table;
                // if (factor_coul < 1.0f) {
                //   const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                //   const float prefactor = qtmp * ((float)q[j]) * table;
                //   ecoul -= (1.0f-factor_coul)*prefactor;
                // }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * ((float)q[j]) / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}


template<class DeviceType, int NEIGHFLAG, int EVFLAG, int LJ_TABLE, int COUL_TABLE>
__global__ void float_force_kernel_expr_performance_float(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_floatq* x_floatq, typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  __half2 *coul_ftable, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f,
  float *d_rtable_f, float *d_drtable_f, float *d_ftable_f, float *d_dftable_f,
  float *d_ctable_f, float *d_dctable_f, float *d_etable_f, float *d_detable_f) {

  __shared__ __half2 lj_ftable_shared[((R_BND + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS) + 1];
  __shared__ __half2 coul_ftable_shared[((R_BND + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS) + 1];
  //__shared__ float2 lj_param_table_shared[TYPE_DIM * TYPE_DIM];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];
  //__shared__ float2 lj_param_table_upper_shared[TYPE_DIM * (TYPE_DIM + 1) / 2];

  for(ushort i = (LJ_LBND >> LJ_BITS) + threadIdx.x; i <= ((R_BND + LJ_OFFSET) >> LJ_BITS); i += blockDim.x) {
    lj_ftable_shared[i - (LJ_LBND >> LJ_BITS)] = lj_ftable[i << LJ_BITS];
  }
  for(ushort i = (COUL_LBND >> COUL_BITS) + threadIdx.x; i <= ((R_BND + COUL_OFFSET) >> COUL_BITS); i += blockDim.x) {
    coul_ftable_shared[i - (COUL_LBND >> COUL_BITS)] = coul_ftable[i << COUL_BITS];
  }
  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  // int param_bnd = ntypes * (ntypes + 1) / 2;
  // for(int i = threadIdx.x; i < param_bnd; i += blockDim.x) {
  //   lj_param_table_upper_shared[i] = lj_param_table_upper_f[i];
  // }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];
    const int itype = (int)(x_data_i.type);
    //const float qtmp = __half2float(x_data_i.q);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    //const float qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    // if(i == 100) {
    //   const int neigh_stride = &d_neighbors(i,1)-&d_neighbors(i,0);
    //   printf("neigh stride : %d\n", neigh_stride);
    // }

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      // const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      // const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      //const float qj = __half2float(x_data_j.q);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      //const float qj = q(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      // if (rsq < ((float)d_cutsq(itype,jtype))) {
      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        //if (rsq < (float)d_cut_ljsq(itype,jtype)) {
        if (rsq < cut_ljsq_f) {
          if (LJ_TABLE) {
            /// float with table, half in half out, params.lj from table
            const float r2inv = 1.0f/rsq;
            const float r6inv = r2inv*r2inv*r2inv;
            float forcelj, switch1, switch2, englj;
            float2 lj_param = lj_param_table_upper_f[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4
            forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);
            if (rsq > ((float)cut_lj_innersq)) {
              __half2 etable = lj_ftable_shared[((__half_as_ushort(__float2half(rsq)) + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS)];
              englj = (lj_param.x * r6inv - lj_param.y);
              forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
            }
            fpair+=factor_lj*forcelj*r2inv;
          }
          else {
            // total float
            const float r2inv = 1.0f/rsq;
            const float r6inv = r2inv*r2inv*r2inv;
            float forcelj, switch1, switch2, englj;
            float2 lj_param = lj_param_table_upper_f[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4

            forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

            if (rsq > ((float)cut_lj_innersq)) {
              switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                        (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
              switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
              englj = r6inv * (lj_param.x * r6inv - lj_param.y);
              forcelj = forcelj*switch1 + englj*switch2;
            }
            fpair+=factor_lj*forcelj*r2inv;
          }
        }
        // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
              if (COUL_TABLE) {  
                /// float with table, half in half out
                ushort itable = __half_as_ushort(__float2half(rsq));
                //__half2 ftable = coul_ftable[itable];
                //__half2 ftable = coul_ftable_shared[itable - COUL_LBND];
                __half2 ftable = coul_ftable_shared[((itable + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS)];
                float forcecoul = __half2float(ftable.x);
                if (factor_coul < 1.0) {
                  forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
                }
                fpair += qtmp * qj * forcecoul;
              }
              else {
                /// total float
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
                float forcecoul = qtmp* qj * table;
                if (factor_coul < 1.0) {
                  const float table = d_ctable_f[itable] + fraction* d_dctable_f[itable];
                  const float prefactor = qtmp* qj * table;
                  forcecoul -= (1.0f-factor_coul)*prefactor;
                }
                fpair += forcecoul/rsq;
              }

            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            // if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
            if (rsq < cut_ljsq_f) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {

                if (COUL_TABLE) {
                  /// float with table, half in float out
                  ushort itable = __half_as_ushort(__float2half(rsq));
                  float2 etable = coul_etable_f[itable];
                  ecoul = etable.x;
                  if (factor_coul < 1.0) {
                    ecoul -= (1.0f-factor_coul) * etable.y;
                  }
                  ecoul *= qtmp * qj; 
                }
                else {
                  /// total float 
                  union_int_float_t rsq_lookup;
                  rsq_lookup.f = rsq;
                  const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                  const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                  const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                  ecoul = qtmp* ((float)q[j]) * table;
                  if (factor_coul < 1.0f) {
                    const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                    const float prefactor = qtmp * ((float)q[j]) * table;
                    ecoul -= (1.0f-factor_coul)*prefactor;
                  }
                }

              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}


template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_expr_performance_half_shared_mem(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_floatq* x_floatq, typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  __half2 *coul_ftable, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f) {

  __shared__ __half2 lj_ftable_shared[((R_BND + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS) + 1];
  __shared__ __half2 coul_ftable_shared[((R_BND + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS) + 1];
  //__shared__ float2 lj_param_table_shared[TYPE_DIM * TYPE_DIM];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];
  //__shared__ float2 lj_param_table_upper_shared[TYPE_DIM * (TYPE_DIM + 1) / 2];

  for(ushort i = (LJ_LBND >> LJ_BITS) + threadIdx.x; i <= ((R_BND + LJ_OFFSET) >> LJ_BITS); i += blockDim.x) {
    lj_ftable_shared[i - (LJ_LBND >> LJ_BITS)] = lj_ftable[i << LJ_BITS];
  }
  for(ushort i = (COUL_LBND >> COUL_BITS) + threadIdx.x; i <= ((R_BND + COUL_OFFSET) >> COUL_BITS); i += blockDim.x) {
    coul_ftable_shared[i - (COUL_LBND >> COUL_BITS)] = coul_ftable[i << COUL_BITS];
  }
  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  // int param_bnd = ntypes * (ntypes + 1) / 2;
  // for(int i = threadIdx.x; i < param_bnd; i += blockDim.x) {
  //   lj_param_table_upper_shared[i] = lj_param_table_upper_f[i];
  // }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    EV_FLOAT ev;

    const int i = d_ilist(ii);
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];
    const int itype = (int)(x_data_i.type);
    //const float qtmp = __half2float(x_data_i.q);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    //const float qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    // if(i == 100) {
    //   const int neigh_stride = &d_neighbors(i,1)-&d_neighbors(i,0);
    //   printf("neigh stride : %d\n", neigh_stride);
    // }

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      // const float factor_lj = (float)special.lj[j >> SBBITS & 3];
      // const float factor_coul = (float)special.coul[j >> SBBITS & 3];
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      //const float qj = __half2float(x_data_j.q);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      //const float qj = q(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      // if (rsq < ((float)d_cutsq(itype,jtype))) {
      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        //if (rsq < (float)d_cut_ljsq(itype,jtype)) {
        if (rsq < cut_ljsq_f) {
          /// float with table, half in half out, params.lj from table
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;
          //float2 lj_param = lj_param_table_f[itype * lj_param_dim_size + jtype];  // lj3, lj4
          //float2 lj_param = lj_param_table_shared[min(itype, jtype) * lj_param_dim_size + max(itype, jtype)];  // lj3, lj4
          float2 lj_param = lj_param_table_upper_f[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4
          //float2 lj_param = lj_param_table_upper_shared[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4

          // forcelj = r6inv * (((float)params(itype,jtype).lj1)*r6inv -
          //           ((float)params(itype,jtype).lj2));
          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

          if (rsq > ((float)cut_lj_innersq)) {
            //__half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
            __half2 etable = lj_ftable_shared[((__half_as_ushort(__float2half(rsq)) + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS)];
            englj = (lj_param.x * r6inv - lj_param.y);
            forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
              /// float with table, half in half out
              ushort itable = __half_as_ushort(__float2half(rsq));
              //__half2 ftable = coul_ftable[itable];
              //__half2 ftable = coul_ftable_shared[itable - COUL_LBND];
              __half2 ftable = coul_ftable_shared[((itable + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS)];
              float forcecoul = __half2float(ftable.x);
              if (factor_coul < 1.0) {
                forcecoul -= (1.0f-factor_coul) * __half2float(ftable.y);
              }
              fpair += qtmp * qj * forcecoul;

              /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
              // const float table = ((float)d_ftable[itable]) + fraction* ((float)d_dftable[itable]);
              // float forcecoul = qtmp* ((float)q[j]) * table;
              // if (factor_coul < 1.0) {
              //   const float table = ((float)d_ctable[itable]) + fraction* ((float)d_dctable[itable]);
              //   const float prefactor = qtmp* ((float)q[j]) * table;
              //   forcecoul -= (1.0f-factor_coul)*prefactor;
              // }
              // fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);
              if (factor_coul < 1.0f) forcecoul -= (1.0f - factor_coul) * prefactor;

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        // a_f(j,0) -= delx * fpair;
        // a_f(j,1) -= dely * fpair;
        // a_f(j,2) -= delz * fpair;
        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
            // if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
            if (rsq < cut_ljsq_f) {
              const float r2inv = 1.0f / rsq;
              const float r6inv = r2inv * r2inv * r2inv;
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
            // if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {

                /// float with table, half in float out
                ushort itable = __half_as_ushort(__float2half(rsq));
                float2 etable = coul_etable_f[itable];
                ecoul = etable.x;
                if (factor_coul < 1.0) {
                  ecoul -= (1.0f-factor_coul) * etable.y;
                }
                ecoul *= qtmp * qj; 


                /// total float 
                // union_int_float_t rsq_lookup;
                // rsq_lookup.f = rsq;
                // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                // const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                // const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                // ecoul = qtmp* ((float)q[j]) * table;
                // if (factor_coul < 1.0f) {
                //   const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                //   const float prefactor = qtmp * ((float)q[j]) * table;
                //   ecoul -= (1.0f-factor_coul)*prefactor;
                // }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f - factor_coul) * prefactor;
              }
              ev.ecoul += 1.0f * ecoul;
            }
          }

          if (vflag_either) {
            const float v0 = delx*delx*fpair;
            const float v1 = dely*dely*fpair;
            const float v2 = delz*delz*fpair;
            const float v3 = delx*dely*fpair;
            const float v4 = delx*delz*fpair;
            const float v5 = dely*delz*fpair;

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
    // a_f(i,0) += fxtmp;
    // a_f(i,1) += fytmp;
    // a_f(i,2) += fztmp;
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}


template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int ZEROFLAG = 0, class Specialisation = void>
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

  template<int EVFLAG>
  void do_launch(int ntotal, PairStyle* fpair) {
    int threadsPerBlock = 512;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    // float_force_kernel_half_table_no_dtable_shared_mem_aos<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     //c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //     (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f);

    // float_force_kernel_half_table_no_dtable_shared_mem<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //     (c.atom)->ntypes + 1, c.lj_param_table_f);

    // float_force_kernel_half_table_no_dtable<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //     (c.atom)->ntypes + 1, c.lj_param_table_f);

    // float_force_kernel_half_table<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    // float_force_kernel_sim_half_table<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    // float_force_kernel<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    // float_force_kernel_expr_half_table_comp<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, 
    //     c.lj_ftable, c.lj_ftable_f, 
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    // float_force_kernel_expr_part_comp<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, 
    //     c.coul_ftable, c.coul_ftable_f, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //     (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    // float_force_kernel_expr_performance_half_shared_mem<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //     (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f);

    // float_force_kernel_expr_performance_float<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //     c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //     (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //     c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);    

#define LAUNCH_FLOAT_FORCE_KERNEL_EXPR_PERFORMANCE_FLOAT(LJ_TABLE, COUL_TABLE) \
    do {  \
    float_force_kernel_expr_performance_float<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, LJ_TABLE, COUL_TABLE><<<blocksPerGrid, threadsPerBlock>>>( \
        ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f, \
        SpecialVal(c.special_coul, c.special_lj), \
        (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params,  \
        c.ev_array, c.eflag, c.vflag_either, c.vflag_global,  \
        (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
        c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f,  \
        (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f, \
        c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f); \
    } while(0) 

    if (fpair -> method_type == 0) {
      LAUNCH_FLOAT_FORCE_KERNEL_EXPR_PERFORMANCE_FLOAT(0, 0);
    }
    else if (fpair -> method_type == 1) {
      LAUNCH_FLOAT_FORCE_KERNEL_EXPR_PERFORMANCE_FLOAT(0, 1);
    }
    else if (fpair -> method_type == 2) {
      LAUNCH_FLOAT_FORCE_KERNEL_EXPR_PERFORMANCE_FLOAT(1, 0);
    }
    else if (fpair -> method_type == 3) {
      LAUNCH_FLOAT_FORCE_KERNEL_EXPR_PERFORMANCE_FLOAT(1, 1);
    }
    else {
      printf("ERROR: unknown method type for kernel launch\n");
      exit(1);
    }

#undef LAUNCH_FLOAT_FORCE_KERNEL_EXPR_PERFORMANCE_FLOAT
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
      printf("lazy init x_float\n"); fflush(stdout);
      fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
      fpair -> x_float_allocated = true;
      c.x_float = fpair -> x_float;
      printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
    }
    if(fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_floatq\n"); fflush(stdout);
      if(fpair -> x_floatq_size > 0) {
        cudaFree(fpair -> x_floatq);
      }
      cudaMalloc((void**)&(fpair -> x_floatq), (fpair -> x).extent(0) * sizeof(AoS_floatq));
      fpair -> x_floatq_size = (fpair -> x).extent(0);
      c.x_floatq = fpair -> x_floatq;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }
    
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float = c.x_float;
    typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;
    typename ArrayTypes<device_type>::t_int_1d_randomread curr_type = c.type;
    typename ArrayTypes<device_type>::t_float_1d_randomread curr_q = c.q;
    AoS_floatq* curr_x_floatq = c.x_floatq;

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      curr_x_float(i,0) = (float)(curr_x(i,0));
      curr_x_float(i,1) = (float)(curr_x(i,1));
      curr_x_float(i,2) = (float)(curr_x(i,2));

      AoS_floatq temp;
      temp.x[0] = (float)(curr_x(i,0));
      temp.x[1] = (float)(curr_x(i,1));
      temp.x[2] = (float)(curr_x(i,2));
      temp.type = (short)curr_type(i);
      temp.q = (short)(curr_q(i) * Q_FACTOR);
      curr_x_floatq[i] = temp;
    });
    Kokkos::fence();

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    do_launch<0>(ntotal, fpair);
    cudaDeviceSynchronize();

    fpair->cuda_kernel_time += cuda_kernel_timer.seconds();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
      printf("lazy init x_float\n"); fflush(stdout);
      fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
      fpair -> x_float_allocated = true;
      c.x_float = fpair -> x_float;
      printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
    }
    if(fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_floatq\n"); fflush(stdout);
      if(fpair -> x_floatq_size > 0) {
        cudaFree(fpair -> x_floatq);
      }
      cudaMalloc((void**)&(fpair -> x_floatq), (fpair -> x).extent(0) * sizeof(AoS_floatq));
      fpair -> x_floatq_size = (fpair -> x).extent(0);
      c.x_floatq = fpair -> x_floatq;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }
    
    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float = c.x_float;
    typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;
    typename ArrayTypes<device_type>::t_int_1d_randomread curr_type = c.type;
    typename ArrayTypes<device_type>::t_float_1d_randomread curr_q = c.q;
    AoS_floatq* curr_x_floatq = c.x_floatq;

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      curr_x_float(i,0) = (float)(curr_x(i,0));
      curr_x_float(i,1) = (float)(curr_x(i,1));
      curr_x_float(i,2) = (float)(curr_x(i,2));

      AoS_floatq temp;
      temp.x[0] = (float)(curr_x(i,0));
      temp.x[1] = (float)(curr_x(i,1));
      temp.x[2] = (float)(curr_x(i,2));
      temp.type = (short)curr_type(i);
      temp.q = (short)(curr_q(i) * Q_FACTOR);
      curr_x_floatq[i] = temp;
    });
    Kokkos::fence();

    //int threadsPerBlock = 128;
    int threadsPerBlock = 512;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    do_launch<1>(ntotal, fpair);
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


template<class PairStyle, PRECISION_TYPE PRECTYPE, unsigned NEIGHFLAG, int ZEROFLAG = 0, class Specialisation = void>
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
    PairComputeFunctorCustomDouble<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
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
    PairComputeFunctorCustomFloat<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
    //ff.contribute_custom();
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
  ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, HALFTHREAD, 0, Specialisation> (fpair,list);
  return ev;
}

#undef Q_FACTOR

#undef LJ_LBND  
#undef COUL_LBND
#undef R_BND
#undef TYPE_DIM
#undef LJ_BITS
#undef LJ_OFFSET
#undef COUL_BITS
#undef COUL_OFFSET

} // namespace RhodoErfcKernels
