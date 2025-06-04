namespace RhodoErfcKernelsExpr {

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

#define Q_FACTOR 1000
#define Q_NTYPES 90

#define COUL_LBND 0b0100000000000000 // half(2.0)
// #define COUL_RBND 0b0101101000100001 // half(196.0 + eps)
#define COUL_RBND 0b0101101100001001 // half(225.0 + eps)
#define LJ_LBND 0b0101010000000000 // half(64.0)
#define LJ_RBND 0b0101101100001001 // half(225.0 + eps)
#define COUL_BITS  1
#define COUL_OFFSET 1
#define LJ_BITS    2
#define LJ_OFFSET  2

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

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void double_force_kernel_expr_performance_sep_special(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  AoS_doubleq* x_doubleq, typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
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

          fpair+=forcelj*r2inv;
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

              evdwl = englj;
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
                ecoul = qtmp* qj * table;
              } else {
                const F_FLOAT r = sqrtf(rsq);
                const F_FLOAT grij = g_ewald * r;
                const F_FLOAT expm2 = expf(-grij*grij);
                const F_FLOAT t = 1.0 / (1.0 + EWALD_P * grij);
                const F_FLOAT erfc = t * (A1 + t * (A2 + t * (A3 + t * (A4 + t * A5)))) * expm2;
                const F_FLOAT prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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
    
    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
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
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0) {
                  const F_FLOAT table = d_ctable[itable] + fraction * d_dctable[itable];
                  const F_FLOAT prefactor = qtmp * qj * table;
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

  template<int EVFLAG>
  void do_launch(int ntotal, PairStyle* fpair) {

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (fpair -> use_sep_sepcial) {
      double_force_kernel_expr_performance_sep_special<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, list.d_numneigh_special, list.d_neighbors_special, 
          c.x_doubleq, f, SpecialVal(c.special_coul, c.special_lj),
          c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
          c.ncoulmask, c.ncoulshiftbits, c.g_ewald, 
          (c.atom)->ntypes, c.lj_param_table, c.lj_param_table_upper,
          c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);
    }
    else {
      double_force_kernel_expr_performance_double<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_doubleq, c.q, f,
          SpecialVal(c.special_coul, c.special_lj),
          c.m_cutsq[1][1], c.m_cut_coulsq[1][1], c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
          c.ncoulmask, c.ncoulshiftbits, c.g_ewald, 
          (c.atom)->ntypes, c.lj_param_table, c.lj_param_table_upper,
          c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);
    }
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated) {
      // printf("lazy init ev_array\n");
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

    do_launch<0>(ntotal, fpair);

    cudaDeviceSynchronize();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated) {
      // printf("lazy init ev_array\n");
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

    do_launch<1>(ntotal, fpair);

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
__global__ void float_force_kernel_expr_performance(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  AoS_floatq* x_floatq, typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
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

  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
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
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

        atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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
    atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG>
__global__ void float_force_kernel_expr_performance_sep_special(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  AoS_floatq* x_floatq, float* f_float, SpecialVal special,
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

  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
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
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
                /// total float
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
              const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
              float forcecoul = qtmp* qj * table;
              fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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

              evdwl = englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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
    
    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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

    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
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
  KKScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,KKDeviceType,KKScatterSum,DUP> dup_f;

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  KKScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,KKDeviceType,KKScatterSum,DUP> dup_eatom;

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
    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    if (fpair -> use_sep_sepcial) {
      float_force_kernel_expr_performance_sep_special<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, list.d_numneigh_special, list.d_neighbors_special, 
          c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj),
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    }
    else {
      float_force_kernel_expr_performance<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, f,
          SpecialVal(c.special_coul, c.special_lj),
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    }
  }

  void kernel_finalize(int ntotal, PairStyle* fpair) {
    auto curr_f_float = c.f_float;
    double* f_ptr = f.data();

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      f_ptr[i * 3 + 0] += (double)(curr_f_float[i * 3 + 0]);
      f_ptr[i * 3 + 1] += (double)(curr_f_float[i * 3 + 1]);
      f_ptr[i * 3 + 2] += (double)(curr_f_float[i * 3 + 2]);
    });
    Kokkos::fence();
  }
  
  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated|| (fpair -> ev_array).extent(0) < f.extent(0)) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
      // printf("lazy init x_float\n"); fflush(stdout);
      fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
      fpair -> x_float_allocated = true;
      c.x_float = fpair -> x_float;
      // printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
    }
    if(fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      // printf("lazy init x_floatq\n"); fflush(stdout);
      if(fpair -> x_floatq_size > 0) {
        cudaFree(fpair -> x_floatq);
      }
      cudaMalloc((void**)&(fpair -> x_floatq), (fpair -> x).extent(0) * sizeof(AoS_floatq));
      fpair -> x_floatq_size = (fpair -> x).extent(0);
      c.x_floatq = fpair -> x_floatq;
    }

    if (fpair -> f_float_size < f.extent(0)) {
      printf("lazy init f_float\n");
      if (fpair -> f_float_size > 0) {
        cudaFree(fpair -> f_float);
      }
      cudaMalloc((void**)&(fpair -> f_float), f.extent(0) * sizeof(float) * 3);
      fpair -> f_float_size = f.extent(0);
      c.f_float = fpair -> f_float;
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
    float* curr_f_float = c.f_float;

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

      curr_f_float[i * 3 + 0] = 0.0f;
      curr_f_float[i * 3 + 1] = 0.0f;
      curr_f_float[i * 3 + 2] = 0.0f;
    });
    Kokkos::fence();

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    do_launch<0>(ntotal, fpair);
    cudaDeviceSynchronize();

    fpair->cuda_kernel_time += cuda_kernel_timer.seconds();

    kernel_finalize(ntotal, fpair);
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
      // printf("lazy init x_float\n"); fflush(stdout);
      fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
      fpair -> x_float_allocated = true;
      c.x_float = fpair -> x_float;
      // printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
    }
    if(fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      // printf("lazy init x_floatq\n"); fflush(stdout);
      if(fpair -> x_floatq_size > 0) {
        cudaFree(fpair -> x_floatq);
      }
      cudaMalloc((void**)&(fpair -> x_floatq), (fpair -> x).extent(0) * sizeof(AoS_floatq));
      fpair -> x_floatq_size = (fpair -> x).extent(0);
      c.x_floatq = fpair -> x_floatq;
    }

    if (fpair -> f_float_size < f.extent(0)) {
      printf("lazy init f_float\n");
      if (fpair -> f_float_size > 0) {
        cudaFree(fpair -> f_float);
      }
      cudaMalloc((void**)&(fpair -> f_float), f.extent(0) * sizeof(float) * 3);
      fpair -> f_float_size = f.extent(0);
      c.f_float = fpair -> f_float;
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
    float* curr_f_float = c.f_float;

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

      curr_f_float[i * 3 + 0] = 0.0f;
      curr_f_float[i * 3 + 1] = 0.0f;
      curr_f_float[i * 3 + 2] = 0.0f;
    });
    Kokkos::fence();

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    do_launch<1>(ntotal, fpair);
    cudaDeviceSynchronize();

    kernel_finalize(ntotal, fpair);

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


template<class DeviceType, int NEIGHFLAG, int EVFLAG, int USE_RELATIVE_COORD>
__global__ void hfmix_force_kernel_expr_performance_sep_special(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  float binsizex, float binsizey, float binsizez,
  __half2 *coul_ftable, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f,
  float *d_rtable_f, float *d_drtable_f, float *d_ftable_f, float *d_dftable_f,
  float *d_ctable_f, float *d_dctable_f, float *d_etable_f, float *d_detable_f) {

  __shared__ float q_val_arr_shared[Q_NTYPES];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  for(int i = threadIdx.x; i < q_val_num; i += blockDim.x) {
    q_val_arr_shared[i] = q_val_arr[i];
  }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int i = -1, fhcut_num = 0;
  if (ii < ntotal) {
    i = d_ilist(ii);
    fhcut_num = fhcut_split(i);
  }

  // set fhcut_num to the max fhcut_split value 
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
    fhcut_num = max(fhcut_num, other);
  }

  if (ii < ntotal) {
    EV_FLOAT ev;

    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = (int)(x_data_i.type);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (jj >= fhcut_num) {
        const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
        AoS_halfq x_data_j_h;
        uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
        *target_ptr = aligned_ptr[0];
        // AoS_halfq x_data_j_h = x_halfq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
        dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
        delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j_h.type;
        qj = q_val_arr_shared[x_data_j_h.q_type];
      }
      else {
        AoS_floatq x_data_j = x_floatq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
        dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
        delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
        
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j.type;
        qj = ((float)x_data_j.q) / Q_FACTOR;
      }

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
                /// total float
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
              const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
              float forcecoul = qtmp* qj * table;
              fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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

              evdwl = englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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

    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG, int USE_RELATIVE_COORD>
__global__ void hfmix_force_kernel_expr_performance_coul_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  float binsizex, float binsizey, float binsizez,
  __half2 *coul_ftable, float2 *coul_ftable_f, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f,
  float *d_rtable_f, float *d_drtable_f, float *d_ftable_f, float *d_dftable_f,
  float *d_ctable_f, float *d_dctable_f, float *d_etable_f, float *d_detable_f) {

  __shared__ float q_val_arr_shared[Q_NTYPES];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];
  __shared__ __half coul_ftable_shared[((COUL_RBND + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS) + 1];
  // __shared__ float coul_ftable_shared_f[COUL_RBND - COUL_LBND + 1];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  for(int i = threadIdx.x; i < q_val_num; i += blockDim.x) {
    q_val_arr_shared[i] = q_val_arr[i];
  }

  for(ushort i = (COUL_LBND >> COUL_BITS) + threadIdx.x; i <= ((COUL_RBND + COUL_OFFSET) >> COUL_BITS); i += blockDim.x) {
    coul_ftable_shared[i - (COUL_LBND >> COUL_BITS)] = coul_ftable[i << COUL_BITS].x;
  }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int i = -1, fhcut_num = 0;
  if (ii < ntotal) {
    i = d_ilist(ii);
    fhcut_num = fhcut_split(i);
  }

  // set fhcut_num to the max fhcut_split value 
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
    fhcut_num = max(fhcut_num, other);
  }

  if (ii < ntotal) {
    EV_FLOAT ev;

    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = (int)(x_data_i.type);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (jj >= fhcut_num) {
        const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
        AoS_halfq x_data_j_h;
        uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
        *target_ptr = aligned_ptr[0];
        // AoS_halfq x_data_j_h = x_halfq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
        dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
        delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j_h.type;
        qj = q_val_arr_shared[x_data_j_h.q_type];
      }
      else {
        AoS_floatq x_data_j = x_floatq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
        dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
        delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
        
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j.type;
        qj = ((float)x_data_j.q) / Q_FACTOR;
      }

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
                /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
              // const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
              // float forcecoul = qtmp* qj * table;
              // fpair += forcecoul/rsq;

              /// float with table, half in half out
              ushort itable = __half_as_ushort(__float2half(rsq));
              __half ftable = coul_ftable_shared[((itable + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS)];
              fpair += qtmp* qj * __half2float(ftable);

            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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

              evdwl = englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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

    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG, int USE_RELATIVE_COORD>
__global__ void hfmix_force_kernel_expr_performance_coul_table_global(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  float binsizex, float binsizey, float binsizez,
  __half2 *coul_ftable, __half *coul_ftable_x, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f,
  float *d_rtable_f, float *d_drtable_f, float *d_ftable_f, float *d_dftable_f,
  float *d_ctable_f, float *d_dctable_f, float *d_etable_f, float *d_detable_f) {

  __shared__ float q_val_arr_shared[Q_NTYPES];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];
  // __shared__ __half coul_ftable_shared[((COUL_RBND + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS) + 1];
  // __shared__ float coul_ftable_shared_f[COUL_RBND - COUL_LBND + 1];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  for(int i = threadIdx.x; i < q_val_num; i += blockDim.x) {
    q_val_arr_shared[i] = q_val_arr[i];
  }

  // for(ushort i = (COUL_LBND >> COUL_BITS) + threadIdx.x; i <= ((COUL_RBND + COUL_OFFSET) >> COUL_BITS); i += blockDim.x) {
  //   coul_ftable_shared[i - (COUL_LBND >> COUL_BITS)] = coul_ftable[i << COUL_BITS].x;
  // }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int i = -1, fhcut_num = 0;
  if (ii < ntotal) {
    i = d_ilist(ii);
    fhcut_num = fhcut_split(i);
  }

  // set fhcut_num to the max fhcut_split value 
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
    fhcut_num = max(fhcut_num, other);
  }

  if (ii < ntotal) {
    EV_FLOAT ev;

    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = (int)(x_data_i.type);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (jj >= fhcut_num) {
        const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
        AoS_halfq x_data_j_h;
        uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
        *target_ptr = aligned_ptr[0];
        // AoS_halfq x_data_j_h = x_halfq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
        dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
        delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j_h.type;
        qj = q_val_arr_shared[x_data_j_h.q_type];
      }
      else {
        AoS_floatq x_data_j = x_floatq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
        dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
        delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
        
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j.type;
        qj = ((float)x_data_j.q) / Q_FACTOR;
      }

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
                /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
              // const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
              // float forcecoul = qtmp* qj * table;
              // fpair += forcecoul/rsq;

              /// float with table, half in half out
              // ushort itable = __half_as_ushort(__float2half(rsq));
              // __half ftable = coul_ftable_shared[((itable + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS)];
              // fpair += qtmp* qj * __half2float(ftable);
              
              /// float with table, half in half out
              ushort itable = __half_as_ushort(__float2half(rsq));
              __half ftable = coul_ftable_x[itable];
              // __half ftable = coul_ftable[itable].x;
              fpair += qtmp* qj * __half2float(ftable);

            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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

              evdwl = englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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

    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG, int USE_RELATIVE_COORD>
__global__ void hfmix_force_kernel_expr_performance_lj_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  float binsizex, float binsizey, float binsizez,
  __half2 *coul_ftable, float2 *coul_ftable_f, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f,
  float *d_rtable_f, float *d_drtable_f, float *d_ftable_f, float *d_dftable_f,
  float *d_ctable_f, float *d_dctable_f, float *d_etable_f, float *d_detable_f) {

  __shared__ float q_val_arr_shared[Q_NTYPES];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];
  // __shared__ __half coul_ftable_shared[((COUL_RBND + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS) + 1];
  __shared__ __half2 lj_ftable_shared[((LJ_RBND + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS) + 1];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  for(int i = threadIdx.x; i < q_val_num; i += blockDim.x) {
    q_val_arr_shared[i] = q_val_arr[i];
  }
  for(ushort i = (LJ_LBND >> LJ_BITS) + threadIdx.x; i <= ((LJ_RBND + LJ_OFFSET) >> LJ_BITS); i += blockDim.x) {
    lj_ftable_shared[i - (LJ_LBND >> LJ_BITS)] = lj_ftable[i << LJ_BITS];
  }
  // for(ushort i = (COUL_LBND >> COUL_BITS) + threadIdx.x; i <= ((COUL_RBND + COUL_OFFSET) >> COUL_BITS); i += blockDim.x) {
  //   coul_ftable_shared[i - (COUL_LBND >> COUL_BITS)] = coul_ftable[i << COUL_BITS].x;
  // }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int i = -1, fhcut_num = 0;
  if (ii < ntotal) {
    i = d_ilist(ii);
    fhcut_num = fhcut_split(i);
  }

  // set fhcut_num to the max fhcut_split value 
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
    fhcut_num = max(fhcut_num, other);
  }

  if (ii < ntotal) {
    EV_FLOAT ev;

    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = (int)(x_data_i.type);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (jj >= fhcut_num) {
        const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
        AoS_halfq x_data_j_h;
        uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
        *target_ptr = aligned_ptr[0];
        // AoS_halfq x_data_j_h = x_halfq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
        dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
        delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j_h.type;
        qj = q_val_arr_shared[x_data_j_h.q_type];
      }
      else {
        AoS_floatq x_data_j = x_floatq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
        dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
        delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
        
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j.type;
        qj = ((float)x_data_j.q) / Q_FACTOR;
      }

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
          // total float
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;
          float2 lj_param = lj_param_table_upper_f[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4

          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

          if (rsq > ((float)cut_lj_innersq)) {
            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv * (lj_param.x * r6inv - lj_param.y);
            // forcelj = forcelj*switch1 + englj*switch2;

            __half2 etable = lj_ftable_shared[((__half_as_ushort(__float2half(rsq)) + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS)];
            englj = (lj_param.x * r6inv - lj_param.y);
            forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);
          }
          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
                /// total float
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
              const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
              float forcecoul = qtmp* qj * table;
              fpair += forcecoul/rsq;

              /// float with table, half in half out
              // ushort itable = __half_as_ushort(__float2half(rsq));
              // __half ftable = coul_ftable_shared[((itable + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS)];
              // fpair += qtmp* qj * __half2float(ftable);

            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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

              evdwl = englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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

    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG, int USE_RELATIVE_COORD>
__global__ void 
// __launch_bounds__(128, 12)
hfmix_force_kernel_expr_performance_lj_coul_table(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  float binsizex, float binsizey, float binsizez,
  __half2 *coul_ftable, float2 *coul_ftable_f, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f, Coul_LJ_entry* coul_lj_table,
  float *d_rtable_f, float *d_drtable_f, float *d_ftable_f, float *d_dftable_f,
  float *d_ctable_f, float *d_dctable_f, float *d_etable_f, float *d_detable_f) {

  __shared__ float q_val_arr_shared[Q_NTYPES];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];
  // __shared__ __half coul_ftable_shared[((COUL_RBND + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS) + 1];
  // __shared__ __half2 lj_ftable_shared[((LJ_RBND + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS) + 1];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  for(int i = threadIdx.x; i < q_val_num; i += blockDim.x) {
    q_val_arr_shared[i] = q_val_arr[i];
  }
  // for(ushort i = (LJ_LBND >> LJ_BITS) + threadIdx.x; i <= ((LJ_RBND + LJ_OFFSET) >> LJ_BITS); i += blockDim.x) {
  //   lj_ftable_shared[i - (LJ_LBND >> LJ_BITS)] = lj_ftable[i << LJ_BITS];
  // }
  // for(ushort i = (COUL_LBND >> COUL_BITS) + threadIdx.x; i <= ((COUL_RBND + COUL_OFFSET) >> COUL_BITS); i += blockDim.x) {
  //   coul_ftable_shared[i - (COUL_LBND >> COUL_BITS)] = coul_ftable[i << COUL_BITS].x;
  // }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int i = -1, fhcut_num = 0;
  if (ii < ntotal) {
    i = d_ilist(ii);
    fhcut_num = fhcut_split(i);
  }

  // set fhcut_num to the max fhcut_split value 
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
    fhcut_num = max(fhcut_num, other);
  }

  if (ii < ntotal) {
    EV_FLOAT ev;

    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = (int)(x_data_i.type);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (jj >= fhcut_num) {
        const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
        AoS_halfq x_data_j_h;
        uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
        *target_ptr = aligned_ptr[0];
        // AoS_halfq x_data_j_h = x_halfq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
        dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
        delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j_h.type;
        qj = q_val_arr_shared[x_data_j_h.q_type];
      }
      else {
        AoS_floatq x_data_j = x_floatq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
        dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
        delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
        
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j.type;
        qj = ((float)x_data_j.q) / Q_FACTOR;
      }

      if (rsq < cutsq_f) {

        float fpair = 0.0f;
        Coul_LJ_entry table_entry;
        if (rsq > tabinnersq) {
          ushort itable = __half_as_ushort(__float2half(rsq));
          table_entry = coul_lj_table[itable];
        }

        if (rsq < cut_ljsq_f) {
          // total float
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;
          float2 lj_param = lj_param_table_upper_f[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4

          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);



          if (rsq > ((float)cut_lj_innersq)) {
            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv * (lj_param.x * r6inv - lj_param.y);
            // forcelj = forcelj*switch1 + englj*switch2;

            // __half2 etable = lj_ftable_shared[((__half_as_ushort(__float2half(rsq)) + LJ_OFFSET) >> LJ_BITS) - (LJ_LBND >> LJ_BITS)];
            // englj = (lj_param.x * r6inv - lj_param.y);
            // forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

            englj = (lj_param.x * r6inv - lj_param.y);
            forcelj = forcelj * __half2float(table_entry.lj_val.x) + englj * __half2float(table_entry.lj_val.y);
          }
          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
                /// total float
              // union_int_float_t rsq_lookup;
              // rsq_lookup.f = rsq;
              // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              // const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
              // const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
              // float forcecoul = qtmp* qj * table;
              // fpair += forcecoul/rsq;

              /// float with table, half in half out
              // ushort itable = __half_as_ushort(__float2half(rsq));
              // __half ftable = coul_ftable_shared[((itable + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS)];
              // fpair += qtmp* qj * __half2float(ftable);

              fpair += qtmp* qj * table_entry.coul_val;

            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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

              evdwl = englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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

          //   __half2 etable = lj_ftable[__half_as_ushort(__float2half(rsq))];
          //   englj = (lj_param.x * r6inv - lj_param.y);
          //   forcelj = forcelj * __half2float(etable.x) + englj * __half2float(etable.y);

          }
          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

                // ushort itable = __half_as_ushort(__float2half(rsq));
                // float2 ftable = coul_ftable_f[itable];
                // float forcecoul = ftable.x;
                // if (factor_coul < 1.0) {
                //   forcecoul -= (1.0f-factor_coul) * ftable.y;
                // }
                // fpair += qtmp * qj * forcecoul;
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

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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

    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}

template<class DeviceType, int NEIGHFLAG, int EVFLAG, int USE_RELATIVE_COORD>
__global__ void hfmix_force_kernel_expr_performance_half_calc(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cutsq_f, float cut_ljsq_f, float cut_coulsq_f,
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  int eflag, int vflag_either, int vflag_global, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  int ncoulmask, int ncoulshiftbits, float g_ewald, 
  float binsizex, float binsizey, float binsizez,
  __half2 *coul_ftable, float2 *coul_etable_f, __half2 *lj_ftable, float2 *lj_ftable_f, 
  int ntypes, float2 *lj_param_table_f, float2 *lj_param_table_upper_f,
  float *d_rtable_f, float *d_drtable_f, float *d_ftable_f, float *d_dftable_f,
  float *d_ctable_f, float *d_dctable_f, float *d_etable_f, float *d_detable_f) {

  __shared__ float q_val_arr_shared[Q_NTYPES];
  __shared__ float special_lj_shared[4];
  __shared__ float special_coul_shared[4];

  if(threadIdx.x == 0) {
    for(int i = 0; i < 4; i++) {
      special_lj_shared[i] = (float)special.lj[i];
      special_coul_shared[i] = (float)special.coul[i];
    }
  }

  for(int i = threadIdx.x; i < q_val_num; i += blockDim.x) {
    q_val_arr_shared[i] = q_val_arr[i];
  }

  __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int i = -1, fhcut_num = 0;
  if (ii < ntotal) {
    i = d_ilist(ii);
    fhcut_num = fhcut_split(i);
  }

  // set fhcut_num to the max fhcut_split value 
  for (int offset = warpSize / 2; offset > 0; offset /= 2) {
    int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
    fhcut_num = max(fhcut_num, other);
  }

  if (ii < ntotal) {
    EV_FLOAT ev;

    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = (int)(x_data_i.type);
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (jj >= fhcut_num) {
        const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
        AoS_halfq x_data_j_h;
        uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
        *target_ptr = aligned_ptr[0];
        // AoS_halfq x_data_j_h = x_halfq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
        dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
        delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j_h.type;
        qj = q_val_arr_shared[x_data_j_h.q_type];
      }
      else {
        AoS_floatq x_data_j = x_floatq[j];

        delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
        dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
        delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
        
        rsq = delx*delx + dely*dely + delz*delz;
        jtype = x_data_j.type;
        qj = ((float)x_data_j.q) / Q_FACTOR;
      }

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
          // total float
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1, switch2, englj;
          float2 lj_param = lj_param_table_upper_f[((2*ntypes-min(itype, jtype))*(min(itype, jtype)-1)>>1) + (max(itype, jtype)-1)];  // lj3, lj4

          forcelj = r6inv * (12 * lj_param.x * r6inv - 6 * lj_param.y);

          if (rsq > ((float)cut_lj_innersq)) {
            // switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
            //           (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            // switch2 = 12.0f*rsq * (cut_ljsq-rsq) * (rsq-cut_lj_innersq) / denom_lj;
            // englj = r6inv * (lj_param.x * r6inv - lj_param.y);
            // forcelj = forcelj*switch1 + englj*switch2;

            __half switch1_h, switch2_h, englj_h;
            __half rsq_h = __float2half(rsq);
            __half cut_ljsq_h = __float2half(cut_ljsq);
            __half cut_lj_innersq_h = __float2half(cut_lj_innersq);
            __half denom_lj_h = __float2half(denom_lj);
            __half r6inv_h = __float2half(r6inv);
            switch1_h = (cut_ljsq_h-rsq_h) * (cut_ljsq_h-rsq_h) *
                      (cut_ljsq_h + __float2half(2.0f)*rsq_h - __float2half(3.0f)*cut_lj_innersq_h) / denom_lj_h;
            switch2_h = __float2half(12.0f)*rsq_h * (cut_ljsq_h-rsq_h) * (rsq_h-cut_lj_innersq_h) / denom_lj_h;
            englj_h = r6inv_h * (__float2half(lj_param.x) * r6inv_h - __float2half(lj_param.y));
            forcelj = forcelj*__half2float(switch1_h) + __half2float(englj_h*switch2_h);
          }
          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
                /// total float
              union_int_float_t rsq_lookup;
              rsq_lookup.f = rsq;
              const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
              const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
              const float table = d_ftable_f[itable] + fraction* d_dftable_f[itable];
              float forcecoul = qtmp* qj * table;
              fpair += forcecoul/rsq;
            } else {
              const float r = sqrtf(rsq);
              const float grij = g_ewald * r;
              const float expm2 = expf(-grij*grij);
              const float t = 1.0f / (1.0f + EWALD_P_f * grij);
              const float rinv = 1.0f/r;
              const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
              const float prefactor = qqrd2e * qtmp * qj * rinv;
              float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

              fpair += forcecoul * rinv * rinv;
            }
        }

        fxtmp += delx * fpair;
        fytmp += dely * fpair;
        fztmp += delz * fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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

              evdwl = englj;
              ev.evdwl += 1.0f*evdwl;
            }
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp * qj / r;
                ecoul = prefactor * erfc;
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

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      int j = neighbors_i_special(jj);
      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cutsq_f) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq_f) {
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
        if (rsq < cut_coulsq_f) {
            if (rsq > tabinnersq) {
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

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          float evdwl = 0.0f;
          float ecoul = 0.0f;
          if (eflag) {
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
            if (rsq < cut_coulsq_f) {
              if (rsq > tabinnersq) {
                /// total float 
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - d_rtable_f[itable]) * d_drtable_f[itable];
                const float table = d_etable_f[itable] + fraction * d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = d_ctable_f[itable] + fraction * d_dctable_f[itable];
                  const float prefactor = qtmp * qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
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

    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int ZEROFLAG = 0, class Specialisation = void>
struct PairComputeFunctorCustomHfmix  {
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
  KKScatterView<F_FLOAT*[3], typename DAT::t_f_array::array_layout,KKDeviceType,KKScatterSum,DUP> dup_f;

  // The eatom and vatom arrays are atomic for Half/Thread neighbor style
  KKScatterView<E_FLOAT*, typename DAT::t_efloat_1d::array_layout,KKDeviceType,KKScatterSum,DUP> dup_eatom;

  KKScatterView<F_FLOAT*[6], typename DAT::t_virial_array::array_layout,KKDeviceType,KKScatterSum,DUP> dup_vatom;



  NeighListKokkos<device_type> list;

  PairComputeFunctorCustomHfmix(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomHfmix() {c.copymode = 1; list.copymode = 1;};

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

  void kernel_init(int ntotal, PairStyle* fpair) {
    // require use_sep_sepcial for hfmix kernels
    // if (!fpair -> use_sep_sepcial) {
    //   printf("ERROR: require use_sep_sepcial for hfmix kernels\n");
    //   exit(1);
    // }

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
      // printf("lazy init x_float\n"); fflush(stdout);
      fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
      fpair -> x_float_allocated = true;
      c.x_float = fpair -> x_float;
      // printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
    }
    if(fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      // printf("lazy init x_floatq\n"); fflush(stdout);
      if(fpair -> x_floatq_size > 0) {
        cudaFree(fpair -> x_floatq);
        cudaFree(fpair -> x_halfq);
      }
      cudaMalloc((void**)&(fpair -> x_floatq), (fpair -> x).extent(0) * sizeof(AoS_floatq));
      cudaMalloc((void**)&(fpair -> x_halfq), (fpair -> x).extent(0) * sizeof(AoS_halfq));
      fpair -> x_floatq_size = (fpair -> x).extent(0);
      c.x_floatq = fpair -> x_floatq;
      c.x_halfq = fpair -> x_halfq;
    }

    if (fpair -> f_float_size < f.extent(0)) {
      printf("lazy init f_float\n");
      if (fpair -> f_float_size > 0) {
        cudaFree(fpair -> f_float);
      }
      cudaMalloc((void**)&(fpair -> f_float), f.extent(0) * sizeof(float) * 3);
      fpair -> f_float_size = f.extent(0);
      c.f_float = fpair -> f_float;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    if (!fpair -> cuda_var_inited) {
      fpair -> cuda_var_inited = true;
      cudaMalloc((void**)&(fpair -> q_val_idx_map), (Q_FACTOR*2 + 1) * sizeof(int));
      cudaMalloc((void**)&(fpair -> q_val_idx_mask), (Q_FACTOR*2 + 1) * sizeof(int));
      cudaMalloc((void**)&(fpair -> q_val_arr), (Q_FACTOR*2 + 1) * sizeof(float));
      // Kokkos::deep_copy(fpair -> q_val_num, 0);
      fpair -> q_val_num = Kokkos::DualView<int*, device_type>("q_val_num", 1);
      (fpair -> q_val_num).h_view(0) = 0;
      (fpair -> q_val_num).modify_host();
      (fpair -> q_val_num).sync_device();
      auto curr_q_val_idx_map = fpair -> q_val_idx_map;
      auto curr_q_val_idx_mask = fpair -> q_val_idx_mask;
      Kokkos::parallel_for(Q_FACTOR*2 + 1, KOKKOS_LAMBDA (const int i) {
        curr_q_val_idx_map[i] = -1;
        curr_q_val_idx_mask[i] = -1;
      });
      c.q_val_idx_map = fpair -> q_val_idx_map;
      c.q_val_idx_mask = fpair -> q_val_idx_mask;
      c.q_val_arr = fpair -> q_val_arr;
      c.q_val_num = fpair -> q_val_num;

      int table_size = 1<<16;
      cudaMalloc((void**)&(fpair -> coul_ftable_x), table_size * sizeof(__half));
      cudaMalloc((void**)&(fpair -> coul_lj_table), table_size * sizeof(Coul_LJ_entry));
      c.coul_ftable_x = fpair -> coul_ftable_x;
      c.coul_lj_table = fpair -> coul_lj_table;

      auto curr_coul_ftable_x = c.coul_ftable_x;
      auto curr_coul_lj_table = c.coul_lj_table;
      auto curr_coul_ftable = c.coul_ftable;
      auto curr_coul_ftable_f = c.coul_ftable_f;
      auto curr_lj_ftable = c.lj_ftable;

      Kokkos::parallel_for(table_size, KOKKOS_LAMBDA (const int i) {
        curr_coul_ftable_x[i] = curr_coul_ftable[i].x;
        curr_coul_lj_table[i].coul_val = curr_coul_ftable_f[i].x;
        curr_coul_lj_table[i].lj_val = curr_lj_ftable[i];
      });
      Kokkos::fence();

      // int cache_table_size = (COUL_RBND - COUL_LBND) * sizeof(Coul_LJ_entry);
      // int device_id;
      // cudaGetDevice(&device_id);
      // cudaDeviceProp prop;
      // cudaGetDeviceProperties(&prop, device_id);
      // size_t size = min(cache_table_size, prop.persistingL2CacheMaxSize);
      // printf("alloc %d bytes of set-aside L2 cache\n", size);
      // cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, size);

      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_coul_table<typename PairStyle::device_type, NEIGHFLAG, 0, 0>, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_coul_table<typename PairStyle::device_type, NEIGHFLAG, 0, 1>, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_coul_table<typename PairStyle::device_type, NEIGHFLAG, 1, 0>, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_coul_table<typename PairStyle::device_type, NEIGHFLAG, 1, 1>, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_lj_table<typename PairStyle::device_type, NEIGHFLAG, 0, 0>, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_lj_table<typename PairStyle::device_type, NEIGHFLAG, 0, 1>, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_lj_table<typename PairStyle::device_type, NEIGHFLAG, 1, 0>, cudaFuncCachePreferL1);
      cudaFuncSetCacheConfig(hfmix_force_kernel_expr_performance_lj_table<typename PairStyle::device_type, NEIGHFLAG, 1, 1>, cudaFuncCachePreferL1);
    }
    
    // cudaStreamAttrValue stream_attribute; 
    // stream_attribute.accessPolicyWindow.base_ptr  = reinterpret_cast<void*>(fpair -> coul_lj_table);
    // int cache_table_size = (COUL_RBND - COUL_LBND) * sizeof(Coul_LJ_entry);
    // stream_attribute.accessPolicyWindow.num_bytes = cache_table_size;
    // stream_attribute.accessPolicyWindow.hitRatio  = 1.0;
    // stream_attribute.accessPolicyWindow.hitProp   = cudaAccessPropertyPersisting;
    // stream_attribute.accessPolicyWindow.missProp  = cudaAccessPropertyStreaming;
    // cudaStreamSetAttribute(0, cudaStreamAttributeAccessPolicyWindow, &stream_attribute);

    Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> curr_x_float = c.x_float;
    typename ArrayTypes<device_type>::t_x_array_randomread curr_x = c.x;
    typename ArrayTypes<device_type>::t_int_1d_randomread curr_type = c.type;
    typename ArrayTypes<device_type>::t_float_1d_randomread curr_q = c.q;
    AoS_floatq* curr_x_floatq = c.x_floatq;
    AoS_halfq* curr_x_halfq = c.x_halfq;
    float* curr_f_float = c.f_float;
    bool curr_use_relative_coord = fpair->use_relative_coord;
    auto curr_bin_base = c.bin_base;
    auto curr_q_val_num = c.q_val_num;
    auto curr_q_val_idx_map = c.q_val_idx_map;
    auto curr_q_val_idx_mask = c.q_val_idx_mask;
    auto curr_q_val_arr = c.q_val_arr;

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      float q_val = curr_q(i);
      int val_scaled = (int)(q_val * Q_FACTOR);
      if (curr_q_val_idx_mask[val_scaled + Q_FACTOR] == -1) {
        if (Kokkos::atomic_fetch_add(&curr_q_val_idx_mask[val_scaled + Q_FACTOR], 1) == -1) {
          int index = Kokkos::atomic_fetch_add(&curr_q_val_num.d_view(0), 1);
          curr_q_val_idx_map[val_scaled + Q_FACTOR] = index;
          curr_q_val_arr[index] = q_val;
        }
      }
    });
    Kokkos::fence();

    curr_q_val_num.modify_device();
    curr_q_val_num.sync_host();

    if (c.q_val_num.h_view(0) > Q_NTYPES) {
      printf("ERROR: q value type larger than Q_NTYPES\n");
      exit(1);
    }

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      curr_x_float(i,0) = (float)(curr_x(i,0));
      curr_x_float(i,1) = (float)(curr_x(i,1));
      curr_x_float(i,2) = (float)(curr_x(i,2));

      AoS_floatq temp;
      if (curr_use_relative_coord) {
        temp.x[0] = (float)(curr_x(i,0) - curr_bin_base(i,0));
        temp.x[1] = (float)(curr_x(i,1) - curr_bin_base(i,1));
        temp.x[2] = (float)(curr_x(i,2) - curr_bin_base(i,2));
      }
      else {
        temp.x[0] = (float)(curr_x(i,0));
        temp.x[1] = (float)(curr_x(i,1));
        temp.x[2] = (float)(curr_x(i,2));
      }
      temp.type = (short)curr_type(i);
      temp.q = (short)(curr_q(i) * Q_FACTOR);
      curr_x_floatq[i] = temp;

      AoS_halfq temp_h;
      temp_h.x[0] = __float2half(temp.x[0]);
      temp_h.x[1] = __float2half(temp.x[1]);
      temp_h.x[2] = __float2half(temp.x[2]);
      temp_h.type = (uint8_t)(curr_type(i));
      int q_val_scaled = (int)(curr_q(i) * Q_FACTOR);
      temp_h.q_type = (uint8_t)(curr_q_val_idx_map[q_val_scaled + Q_FACTOR]);
      curr_x_halfq[i] = temp_h;
    
      curr_f_float[i * 3 + 0] = 0.0f;
      curr_f_float[i * 3 + 1] = 0.0f;
      curr_f_float[i * 3 + 2] = 0.0f;
    });
    Kokkos::fence();
  }

  template<int EVFLAG>
  void do_launch(int ntotal, PairStyle* fpair) {
    int threadsPerBlock = 416;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

#define LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_SPE_SPECIAL(BLOCK_SIZE, USE_RELATIVE_COORD) \
    do { \
      int threadsPerBlock = BLOCK_SIZE; \
      int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock; \
      hfmix_force_kernel_expr_performance_sep_special<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors,  \
          c.x_float, c.fhcut_split,  \
          list.d_numneigh_special, list.d_neighbors_special,  \
          c.q_val_num.h_view(0), c.q_val_arr, \
          c.x_halfq, c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj), \
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params,  \
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global,  \
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
          c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f,  \
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f, \
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f); \
    } while(0)

#define LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE(BLOCK_SIZE, USE_RELATIVE_COORD) \
    do { \
      int threadsPerBlock = BLOCK_SIZE; \
      int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock; \
      hfmix_force_kernel_expr_performance_coul_table<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors,  \
          c.x_float, c.fhcut_split,  \
          list.d_numneigh_special, list.d_neighbors_special,  \
          c.q_val_num.h_view(0), c.q_val_arr, \
          c.x_halfq, c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj), \
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params,  \
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global,  \
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
          c.coul_ftable, c.coul_ftable_f, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f,  \
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f, \
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f); \
    } while(0)

#define LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_LJ_TABLE(BLOCK_SIZE, USE_RELATIVE_COORD) \
    do { \
      int threadsPerBlock = BLOCK_SIZE; \
      int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock; \
      hfmix_force_kernel_expr_performance_lj_table<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors,  \
          c.x_float, c.fhcut_split,  \
          list.d_numneigh_special, list.d_neighbors_special,  \
          c.q_val_num.h_view(0), c.q_val_arr, \
          c.x_halfq, c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj), \
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params,  \
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global,  \
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
          c.coul_ftable, c.coul_ftable_f, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f,  \
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f, \
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f); \
    } while(0)

#define LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE_GLOBAL(BLOCK_SIZE, USE_RELATIVE_COORD) \
    do { \
      int threadsPerBlock = BLOCK_SIZE; \
      int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock; \
      hfmix_force_kernel_expr_performance_coul_table_global<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors,  \
          c.x_float, c.fhcut_split,  \
          list.d_numneigh_special, list.d_neighbors_special,  \
          c.q_val_num.h_view(0), c.q_val_arr, \
          c.x_halfq, c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj), \
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params,  \
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global,  \
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
          c.coul_ftable, c.coul_ftable_x, \
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f, \
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f); \
    } while(0)

#define LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_LJ_COUL_TABLE(BLOCK_SIZE, USE_RELATIVE_COORD) \
    do { \
      int threadsPerBlock = BLOCK_SIZE; \
      int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock; \
      hfmix_force_kernel_expr_performance_lj_coul_table<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors,  \
          c.x_float, c.fhcut_split,  \
          list.d_numneigh_special, list.d_neighbors_special,  \
          c.q_val_num.h_view(0), c.q_val_arr, \
          c.x_halfq, c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj), \
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params,  \
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global,  \
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
          c.coul_ftable, c.coul_ftable_f, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f,  \
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f, c.coul_lj_table, \
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f); \
    } while(0)

#define LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_HALF_CALC(BLOCK_SIZE, USE_RELATIVE_COORD) \
    do { \
      int threadsPerBlock = BLOCK_SIZE; \
      int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock; \
      hfmix_force_kernel_expr_performance_half_calc<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors,  \
          c.x_float, c.fhcut_split,  \
          list.d_numneigh_special, list.d_neighbors_special,  \
          c.q_val_num.h_view(0), c.q_val_arr, \
          c.x_halfq, c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj), \
          (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params,  \
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global,  \
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
          c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f,  \
          (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f, \
          c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f); \
    } while(0)

    if (fpair -> method_type == 0) {
      if (fpair -> use_relative_coord) {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_SPE_SPECIAL(128, 1);
      }
      else {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_SPE_SPECIAL(128, 0);
      }
    }
    else if (fpair -> method_type == 1) {
      if (fpair -> use_relative_coord) {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE(416, 1);
      }
      else {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE(416, 0);
      }
    }
    else if (fpair -> method_type == 2) {
      if (fpair -> use_relative_coord) {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_LJ_TABLE(416, 1);
      }
      else {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_LJ_TABLE(416, 0);
      }
    }
    else if (fpair -> method_type == 3) {
      if (fpair -> use_relative_coord) {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE_GLOBAL(128, 1);
      }
      else {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE_GLOBAL(128, 0);
      }
    }
    else if (fpair -> method_type == 4) {
      if (fpair -> use_relative_coord) {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_LJ_COUL_TABLE(128, 1);
      }
      else {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_LJ_COUL_TABLE(128, 0);
      }
    }
    else if (fpair -> method_type == 5) {
      // use 
      if (fpair -> use_relative_coord) {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_HALF_CALC(128, 1);
      }
      else {
        LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_HALF_CALC(128, 0);
      }
    }

    // else if (fpair -> method_type == 10) {
    //   float_force_kernel_expr_performance_float<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
    //       ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //       SpecialVal(c.special_coul, c.special_lj),
    //       (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //       c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //       (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //       c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //       (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //       c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    // }    
    // else if (fpair -> method_type == 11) {
    //   float_force_kernel_expr_performance_float<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
    //       ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //       SpecialVal(c.special_coul, c.special_lj),
    //       (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //       c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //       (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //       c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //       (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //       c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    // }
    // else if (fpair -> method_type == 12) {
    //   hfmix_force_kernel_expr_performance_coul_table_1<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //       ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, 
    //       c.x_float, c.fhcut_split, 
    //       list.d_numneigh_special, list.d_neighbors_special, 
    //       c.q_val_num.h_view(0), c.q_val_arr,
    //       c.x_halfq, c.x_floatq, c.f_float, SpecialVal(c.special_coul, c.special_lj),
    //       (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //       c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //       (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //       c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, 
    //       (float)c.binsizex, (float)c.binsizey, (float)c.binsizez,
    //       c.coul_ftable, c.coul_ftable_f, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //       (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //       c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    // }
    // else if (fpair -> method_type == 13) {
    //   float_force_kernel_expr_performance_float_coul_table<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //       ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //       SpecialVal(c.special_coul, c.special_lj),
    //       (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //       c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //       (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //       c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //       (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //       c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    // }
    // else if (fpair -> method_type == 14) {
    //   float_force_kernel_expr_performance_float_no_coul_table<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //       ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //       SpecialVal(c.special_coul, c.special_lj),
    //       (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //       c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //       (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //       c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //       (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //       c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    // }
    // else if (fpair -> method_type == 15) {
    //   float_force_kernel_expr_performance_float_no_coul_table_no_shared<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //       ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_floatq, c.q, f,
    //       SpecialVal(c.special_coul, c.special_lj),
    //       (float)c.m_cutsq[1][1], (float)c.m_cut_ljsq[1][1], (float)c.m_cut_coulsq[1][1], c.params, 
    //       c.ev_array, c.eflag, c.vflag_either, c.vflag_global, 
    //       (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //       c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald, c.coul_ftable, c.coul_etable_f, c.lj_ftable, c.lj_ftable_f, 
    //       (c.atom)->ntypes, c.lj_param_table_f, c.lj_param_table_upper_f,
    //       c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);
    // }
    
    else {
      printf("ERROR: unknown method type for kernel launch\n");
      exit(1);
    }

#undef LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_SPE_SPECIAL
#undef LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE
#undef LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_LJ_TABLE
#undef LAUNCH_HFMIX_FORCE_KERNEL_EXPR_PERFORMANCE_COUL_TABLE_GLOBAL
  }

  void kernel_finalize(int ntotal, PairStyle* fpair) {
    auto curr_f_float = c.f_float;
    double* f_ptr = f.data();

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
      f_ptr[i * 3 + 0] += (double)(curr_f_float[i * 3 + 0]);
      f_ptr[i * 3 + 1] += (double)(curr_f_float[i * 3 + 1]);
      f_ptr[i * 3 + 2] += (double)(curr_f_float[i * 3 + 2]);
    });
    Kokkos::fence();
  }
  
  void kernel_launch(int ntotal, PairStyle* fpair) {
    kernel_init(ntotal, fpair);

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    do_launch<0>(ntotal, fpair);
    cudaDeviceSynchronize();

    fpair->cuda_kernel_time += cuda_kernel_timer.seconds();

    kernel_finalize(ntotal, fpair);
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    kernel_init(ntotal, fpair);

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    do_launch<1>(ntotal, fpair);
    cudaDeviceSynchronize();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();

    kernel_finalize(ntotal, fpair);

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();
    
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
    // printf("in RhodoErfcKernelsExpr rhodo double kernel\n");
    PairComputeFunctorCustomDouble<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
  }
  else if(PRECTYPE == FLOAT_PREC) {
    // printf("in RhodoErfcKernelsExpr rhodo float kernel\n");
    PairComputeFunctorCustomFloat<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
  }
  else if(PRECTYPE == HFMIX_PREC) {
    // printf("in RhodoErfcKernelsExpr rhodo hfmix kernel\n");
    PairComputeFunctorCustomHfmix<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
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
  ev = pair_compute_neighlist_custom <PairStyle, PRECTYPE, HALFTHREAD, 0, Specialisation> (fpair,list);
  return ev;
}

#undef Q_FACTOR
#undef Q_NTYPES

#undef COUL_LBND
#undef COUL_RBND
#undef COUL_BITS
#undef COUL_OFFSET


} // namespace RhodoErfcKernelsExpr
