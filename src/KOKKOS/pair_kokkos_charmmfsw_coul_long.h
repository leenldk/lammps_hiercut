namespace CharmmfswCoulLongKernels {

const double EWALD_F =   1.12837917;
const double EWALD_P =   0.3275911;
const double A1      =   0.254829592;
const double A2      =  -0.284496736;
const double A3      =   1.421413741;
const double A4      =  -1.453152027;
const double A5      =   1.061405429;

const float EWALD_F_f =   1.12837917f;
const float EWALD_P_f =   0.3275911f;
const float A1_f      =   0.254829592f;
const float A2_f      =  -0.284496736f;
const float A3_f      =   1.421413741f;
const float A4_f      =  -1.453152027f;
const float A5_f      =   1.061405429f;

#define Q_FACTOR 1000
#define Q_NTYPES 90

#define COUL_LBND 0b0100000000000000 // half(2.0)
// #define COUL_RBND 0b0101101000100001 // half(196.0 + eps)
#define COUL_RBND 0b0101101100001001 // half(225.0 + eps)
#define COUL_BITS  1
#define COUL_OFFSET 1

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
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, int eflag_either, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double tabinnersq,
  double cut_lj3, double cut_lj6, double cut_lj3inv, double cut_lj6inv, double cut_lj_inner3inv, double cut_lj_inner6inv, double denom_lj6, double denom_lj12,
  int ncoulmask, int ncoulshiftbits, double g_ewald,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    const int itype = type(i);
    const F_FLOAT qtmp = q(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),&d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = special.lj[j >> SBBITS & 3];
      const F_FLOAT factor_coul = special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < d_cut_ljsq(itype,jtype)) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1;

          forcelj = r6inv *
            (params(itype,jtype).lj1*r6inv -
            params(itype,jtype).lj2);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
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

        a_f(j,0) -= delx*fpair;
        a_f(j,1) -= dely*fpair;
        a_f(j,2) -= delz*fpair;

        if (EVFLAG) {
          // F_FLOAT evdwl = 0.0;
          if (eflag_either) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              // evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              // ev.evdwl += 1.0*evdwl;
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              const F_FLOAT r = sqrt(rsq);
              const F_FLOAT rinv = 1.0/r;
              const F_FLOAT r3inv = rinv*rinv*rinv;
              F_FLOAT englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = params(itype,jtype).lj3*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -params(itype,jtype).lj4*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj3*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -params(itype,jtype).lj4*r6inv +
                  params(itype,jtype).lj4*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += 1.0 * factor_lj * englj;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              F_FLOAT ecoul = 0.0;
              // ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
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

          // if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl+ecoul,fpair,delx,dely,delz);
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


template<class DeviceType, int NEIGHFLAG, int EVFLAG, int EFLAG_EITHER, int USE_SEP_SPECIAL>
__global__ void double_force_kernel_performance(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  double qqrd2e, double cut_ljsq, double cut_lj_innersq, double denom_lj, double tabinnersq,
  double cut_lj3, double cut_lj6, double cut_lj3inv, double cut_lj6inv, double cut_lj_inner3inv, double cut_lj_inner6inv, double denom_lj6, double denom_lj12,
  int ncoulmask, int ncoulshiftbits, double g_ewald,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

    __shared__ double special_lj_shared[4];
    __shared__ double special_coul_shared[4];
    
    if (threadIdx.x < 4) {
      special_lj_shared[threadIdx.x] = special.lj[threadIdx.x];
      special_coul_shared[threadIdx.x] = special.coul[threadIdx.x];
    }
    __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    // auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const X_FLOAT xtmp = x(i,0);
    const X_FLOAT ytmp = x(i,1);
    const X_FLOAT ztmp = x(i,2);
    const int itype = type(i);
    const F_FLOAT qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),&d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const F_FLOAT factor_lj = special_lj_shared[j >> SBBITS & 3];
      const F_FLOAT factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const X_FLOAT delx = xtmp - x(j,0);
      const X_FLOAT dely = ytmp - x(j,1);
      const X_FLOAT delz = ztmp - x(j,2);
      const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < d_cut_ljsq(itype,jtype)) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1;

          forcelj = r6inv *
            (params(itype,jtype).lj1*r6inv -
            params(itype,jtype).lj2);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < d_cut_coulsq(itype,jtype)) {
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

        // a_f(j,0) -= delx*fpair;
        // a_f(j,1) -= dely*fpair;
        // a_f(j,2) -= delz*fpair;
        atomicAdd(&f_ptr[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_ptr[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_ptr[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < d_cut_ljsq(itype,jtype)) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              const F_FLOAT r = sqrt(rsq);
              const F_FLOAT rinv = 1.0/r;
              const F_FLOAT r3inv = rinv*rinv*rinv;
              F_FLOAT englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = params(itype,jtype).lj3*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -params(itype,jtype).lj4*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*params(itype,jtype).lj3*r6inv -
                params(itype,jtype).lj3*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -params(itype,jtype).lj4*r6inv +
                  params(itype,jtype).lj4*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += 1.0 * factor_lj * englj;
            }
            if (rsq < d_cut_coulsq(itype,jtype)) {
              F_FLOAT ecoul = 0.0;
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
        }
      }
    }
    if (EVFLAG && EFLAG_EITHER) {
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


// template specialization for USE_SEP_SPECIAL=1 case
template<class DeviceType, int NEIGHFLAG, int EVFLAG, int EFLAG_EITHER>
__global__ void double_force_kernel_performance_sep_special(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  AoS_doubleq* x_doubleq,
  typename ArrayTypes<DeviceType>::t_f_array f, SpecialVal special,
  double cut_sq, double cut_ljsq, double cut_coulsq,
  int ntypes, double2 *param_lj12, double2 *param_lj34,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  double qqrd2e, double cut_lj_innersq, double denom_lj, double tabinnersq,
  double cut_lj3, double cut_lj6, double cut_lj3inv, double cut_lj6inv, double cut_lj_inner3inv, double cut_lj_inner6inv, double denom_lj6, double denom_lj12,
  int ncoulmask, int ncoulshiftbits, double g_ewald,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

    // USE_SEP_SPECIAL = 1

    __shared__ double special_lj_shared[4];
    __shared__ double special_coul_shared[4];
    
    if (threadIdx.x < 4) {
      special_lj_shared[threadIdx.x] = special.lj[threadIdx.x];
      special_coul_shared[threadIdx.x] = special.coul[threadIdx.x];
    }
    __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    // auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int i = d_ilist(ii);
    AoS_doubleq x_data_i = x_doubleq[i];
    const X_FLOAT xtmp = x_data_i.x[0];
    const X_FLOAT ytmp = x_data_i.x[1];
    const X_FLOAT ztmp = x_data_i.x[2];
    const int itype = x_data_i.type;
    const double qtmp = ((double)x_data_i.q) / Q_FACTOR;
    // const X_FLOAT xtmp = x(i,0);
    // const X_FLOAT ytmp = x(i,1);
    // const X_FLOAT ztmp = x(i,2);
    // const int itype = type(i);
    // const F_FLOAT qtmp = q(i);
    double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),&d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    F_FLOAT fxtmp = 0.0;
    F_FLOAT fytmp = 0.0;
    F_FLOAT fztmp = 0.0;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      // const F_FLOAT factor_lj = special_lj_shared[j >> SBBITS & 3];
      // const F_FLOAT factor_coul = special_coul_shared[j >> SBBITS & 3];
      // j &= NEIGHMASK;
      AoS_doubleq x_data_j = x_doubleq[j];
      const X_FLOAT delx = xtmp - x_data_j.x[0];
      const X_FLOAT dely = ytmp - x_data_j.x[1];
      const X_FLOAT delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const double qj = ((double)x_data_j.q) / Q_FACTOR;
      // const X_FLOAT delx = xtmp - x(j,0);
      // const X_FLOAT dely = ytmp - x(j,1);
      // const X_FLOAT delz = ztmp - x(j,2);
      // const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        F_FLOAT fpair = F_FLOAT();

        // if (rsq < d_cut_ljsq(itype,jtype)) {
        if (rsq < cut_ljsq) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=forcelj*r2inv;
        }
        // if (rsq < d_cut_coulsq(itype,jtype)) {
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
            const F_FLOAT table = d_ftable[itable] + fraction*d_dftable[itable];
            F_FLOAT forcecoul = qtmp* qj * table;
            // if (factor_coul < 1.0) {
            //   const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
            //   const F_FLOAT prefactor = qtmp*q[j] * table;
            //   forcecoul -= (1.0-factor_coul)*prefactor;
            // }
            fpair += forcecoul/rsq;
          } else {
            const F_FLOAT r = sqrt(rsq);
            const F_FLOAT grij = g_ewald * r;
            const F_FLOAT expm2 = exp(-grij*grij);
            const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
            const F_FLOAT rinv = 1.0/r;
            const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
            const F_FLOAT prefactor = qqrd2e * qtmp* qj *rinv;
            F_FLOAT forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            // if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        // a_f(j,0) -= delx*fpair;
        // a_f(j,1) -= dely*fpair;
        // a_f(j,2) -= delz*fpair;
        atomicAdd(&f_ptr[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_ptr[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_ptr[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              const F_FLOAT r = sqrt(rsq);
              const F_FLOAT rinv = 1.0/r;
              const F_FLOAT r3inv = rinv*rinv*rinv;
              F_FLOAT englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += englj;
            }
            if (rsq < cut_coulsq) {
              F_FLOAT ecoul = 0.0;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
                const F_FLOAT table = d_etable[itable] + fraction*d_detable[itable];
                ecoul = qtmp* qj * table;
                // if (factor_coul < 1.0) {
                //   const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
                //   const F_FLOAT prefactor = qtmp*q[j] * table;
                //   ecoul -= (1.0-factor_coul)*prefactor;
                // }
              } else {
                const F_FLOAT r = sqrt(rsq);
                const F_FLOAT grij = g_ewald * r;
                const F_FLOAT expm2 = exp(-grij*grij);
                const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
                const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const F_FLOAT prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
                // if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
              }
              ev.ecoul += ecoul;
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
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const double qj = ((double)x_data_j.q) / Q_FACTOR;
      // const X_FLOAT delx = xtmp - x(j,0);
      // const X_FLOAT dely = ytmp - x(j,1);
      // const X_FLOAT delz = ztmp - x(j,2);
      // const int jtype = type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < cut_ljsq) {
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
            const F_FLOAT table = d_ftable[itable] + fraction*d_dftable[itable];
            F_FLOAT forcecoul = qtmp* qj * table;
            if (factor_coul < 1.0) {
              const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
              const F_FLOAT prefactor = qtmp* qj * table;
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
            const F_FLOAT prefactor = qqrd2e * qtmp* qj *rinv;
            F_FLOAT forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0) forcecoul -= (1.0-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        // a_f(j,0) -= delx*fpair;
        // a_f(j,1) -= dely*fpair;
        // a_f(j,2) -= delz*fpair;
        atomicAdd(&f_ptr[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_ptr[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_ptr[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              const F_FLOAT r = sqrt(rsq);
              const F_FLOAT rinv = 1.0/r;
              const F_FLOAT r3inv = rinv*rinv*rinv;
              F_FLOAT englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += 1.0 * factor_lj * englj;
            }
            if (rsq < cut_coulsq) {
              F_FLOAT ecoul = 0.0;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const F_FLOAT fraction = (rsq_lookup.f - d_rtable[itable]) * d_drtable[itable];
                const F_FLOAT table = d_etable[itable] + fraction*d_detable[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0) {
                  const F_FLOAT table = d_ctable[itable] + fraction*d_dctable[itable];
                  const F_FLOAT prefactor = qtmp* qj * table;
                  ecoul -= (1.0-factor_coul)*prefactor;
                }
              } else {
                const F_FLOAT r = sqrt(rsq);
                const F_FLOAT grij = g_ewald * r;
                const F_FLOAT expm2 = exp(-grij*grij);
                const F_FLOAT t = 1.0 / (1.0 + EWALD_P*grij);
                const F_FLOAT erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const F_FLOAT prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0) ecoul -= (1.0-factor_coul)*prefactor;
              }
              ev.ecoul += 1.0*ecoul;
            }
          }
        }
      }
    }

    if (EVFLAG && EFLAG_EITHER) {
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

//Specialisation for Neighborlist types Half, HalfThread, Full
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

  PairComputeFunctorCustomDouble(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomDouble() {c.copymode = 1; list.copymode = 1;};

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
      const X_FLOAT delx = xtmp - c.x(j,0);
      const X_FLOAT dely = ytmp - c.x(j,1);
      const X_FLOAT delz = ztmp - c.x(j,2);
      const int jtype = c.type(j);
      const F_FLOAT rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < c.d_cutsq(itype,jtype)) {

        F_FLOAT fpair = F_FLOAT();

        if (rsq < c.d_cut_ljsq(itype,jtype)) {
          // fpair+=factor_lj*c.template compute_fpair<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
          const F_FLOAT r2inv = 1.0/rsq;
          const F_FLOAT r6inv = r2inv*r2inv*r2inv;
          F_FLOAT forcelj, switch1;

          forcelj = r6inv *
            (c.params(itype,jtype).lj1*r6inv -
            c.params(itype,jtype).lj2);

          if (rsq > c.cut_lj_innersq) {
            switch1 = (c.cut_ljsq-rsq) * (c.cut_ljsq-rsq) *
                      (c.cut_ljsq + 2.0*rsq - 3.0*c.cut_lj_innersq) / c.denom_lj;
            forcelj = forcelj*switch1;
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

        a_f(j,0) -= delx*fpair;
        a_f(j,1) -= dely*fpair;
        a_f(j,2) -= delz*fpair;

        if (EVFLAG) {
          F_FLOAT evdwl = 0.0;
          F_FLOAT ecoul = 0.0;
          if (c.eflag_either) {
            if (rsq < c.d_cut_ljsq(itype,jtype)) {
              // evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              // ev.evdwl += 1.0*evdwl;
              const F_FLOAT r2inv = 1.0/rsq;
              const F_FLOAT r6inv = r2inv*r2inv*r2inv;
              const F_FLOAT r = sqrt(rsq);
              const F_FLOAT rinv = 1.0/r;
              const F_FLOAT r3inv = rinv*rinv*rinv;
              F_FLOAT englj, englj12, englj6;

              if (rsq > c.cut_lj_innersq) {
                englj12 = c.params(itype,jtype).lj3*c.cut_lj6*
                  c.denom_lj12 * (r6inv - c.cut_lj6inv)*(r6inv - c.cut_lj6inv);
                englj6 = -c.params(itype,jtype).lj4*
                  c.cut_lj3*c.denom_lj6 * (r3inv - c.cut_lj3inv)*(r3inv - c.cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*c.params(itype,jtype).lj3*r6inv -
                c.params(itype,jtype).lj3*c.cut_lj_inner6inv*c.cut_lj6inv;
                englj6 = -c.params(itype,jtype).lj4*r6inv +
                  c.params(itype,jtype).lj4*c.cut_lj_inner3inv*c.cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += 1.0 * factor_lj * englj;
            }
            if (rsq < c.d_cut_coulsq(itype,jtype)) {
              // ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
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

          // if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl+ecoul,fpair,delx,dely,delz);
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
    // if (c.newton_pair) 
    // compute_item<0,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    compute_item_custom<0,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    // else compute_item<0,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const int i, value_type &energy_virial) const {
    // if (c.newton_pair)
    // energy_virial += compute_item<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    energy_virial += compute_item_custom<1,1>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
    // else
    //   energy_virial += compute_item<1,0>(i,list,typename DoCoul<PairStyle::COUL_FLAG>::type());
  }

  template<int EVFLAG>
  void do_launch(int ntotal, PairStyle* fpair) {

    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;
    
    if (fpair -> reorder_neighbor == ON_NEIGH_BUILD) {
      printf("ERROR: reorder_neighbor not supported for double kernels\n");
      exit(1);
    }

    // double_force_kernel<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, c.q, dup_f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag_either, 
    //     c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
    //     c.cut_lj3, c.cut_lj6, c.cut_lj3inv, c.cut_lj6inv, c.cut_lj_inner3inv, c.cut_lj_inner6inv, c.denom_lj6, c.denom_lj12,
    //     c.ncoulmask, c.ncoulshiftbits, c.g_ewald,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    // printf("d_neighbors_special : %d, %d\n", list.d_neighbors_special.extent(0), list.d_neighbors_special.extent(1));
    // int max_num_special = 0;
    // int total_num_special = 0;
    // typename ArrayTypes<device_type>::t_int_1d curr_numneigh_special = list.d_numneigh_special;
    // Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA(const int i, int& max_num_special, int& total_num_special) {
    //   const int jnum_special = curr_numneigh_special(i);
    //   max_num_special = max(max_num_special, jnum_special);
    //   total_num_special += jnum_special;
    // }, Kokkos::Max<int>(max_num_special), total_num_special);
    // Kokkos::fence();
    // printf("max speial neighbor : %d, total special neighbor : %d\n", max_num_special, total_num_special);

    if (c.eflag_either) {
      if (fpair -> use_sep_sepcial) {
        double_force_kernel_performance_sep_special<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, list.d_numneigh_special, list.d_neighbors_special, 
            c.x_doubleq, f,
            SpecialVal(c.special_coul, c.special_lj),
            c.cut_sq, c.cut_ljsq, c.cut_coulsq, 
            c.atom->ntypes, c.param_lj12, c.param_lj34,  
            c.ev_array,
            c.qqrd2e, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
            c.cut_lj3, c.cut_lj6, c.cut_lj3inv, c.cut_lj6inv, c.cut_lj_inner3inv, c.cut_lj_inner6inv, c.denom_lj6, c.denom_lj12,
            c.ncoulmask, c.ncoulshiftbits, c.g_ewald,
            c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);
      }
      else {
        double_force_kernel_performance<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, 1, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, list.d_numneigh_special, list.d_neighbors_special, 
            c.x, c.type, c.q, f,
            SpecialVal(c.special_coul, c.special_lj),
            c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
            c.ev_array,
            c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
            c.cut_lj3, c.cut_lj6, c.cut_lj3inv, c.cut_lj6inv, c.cut_lj_inner3inv, c.cut_lj_inner6inv, c.denom_lj6, c.denom_lj12,
            c.ncoulmask, c.ncoulshiftbits, c.g_ewald,
            c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);
      }
    }
    else {
      if (fpair -> use_sep_sepcial) {
        double_force_kernel_performance_sep_special<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, list.d_numneigh_special, list.d_neighbors_special, 
            c.x_doubleq, f,
            SpecialVal(c.special_coul, c.special_lj),
            c.cut_sq, c.cut_ljsq, c.cut_coulsq, 
            c.atom->ntypes, c.param_lj12, c.param_lj34,  
            c.ev_array,
            c.qqrd2e, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
            c.cut_lj3, c.cut_lj6, c.cut_lj3inv, c.cut_lj6inv, c.cut_lj_inner3inv, c.cut_lj_inner6inv, c.denom_lj6, c.denom_lj12,
            c.ncoulmask, c.ncoulshiftbits, c.g_ewald,
            c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);
      }
      else {
        double_force_kernel_performance<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, 0, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, list.d_numneigh_special, list.d_neighbors_special, 
            c.x, c.type, c.q, f,
            SpecialVal(c.special_coul, c.special_lj),
            c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
            c.ev_array,
            c.qqrd2e, c.cut_ljsq, c.cut_lj_innersq, c.denom_lj, c.tabinnersq,
            c.cut_lj3, c.cut_lj6, c.cut_lj3inv, c.cut_lj6inv, c.cut_lj_inner3inv, c.cut_lj_inner6inv, c.denom_lj6, c.denom_lj12,
            c.ncoulmask, c.ncoulshiftbits, c.g_ewald,
            c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);
      }
    }
  }
  

  void kernel_launch(int ntotal, PairStyle* fpair) {
    // printf("in kernel_launch\n");    
    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (fpair -> x_doubleq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_doubleq\n");
      if (fpair -> x_doubleq_size > 0) {
        cudaFree(fpair -> x_doubleq);
      }
      cudaMalloc((void**)&(fpair -> x_doubleq), (fpair -> x).extent(0) * sizeof(AoS_doubleq));
      fpair -> x_doubleq_size = (fpair -> x).extent(0);
      c.x_doubleq = fpair -> x_doubleq;
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
    // printf("ntotal value : %d\n", ntotal);
    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (fpair -> x_doubleq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_doubleq\n");
      if (fpair -> x_doubleq_size > 0) {
        cudaFree(fpair -> x_doubleq);
      }
      cudaMalloc((void**)&(fpair -> x_doubleq), (fpair -> x).extent(0) * sizeof(AoS_doubleq));
      fpair -> x_doubleq_size = (fpair -> x).extent(0);
      c.x_doubleq = fpair -> x_doubleq;
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
__global__ void float_force_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d_randomread type,
  typename ArrayTypes<DeviceType>::t_float_1d_randomread q,
  KKScatterView<F_FLOAT*[3], typename ArrayTypes<DeviceType>::t_f_array::array_layout,typename KKDevice<DeviceType>::value,KKScatterSum,typename NeedDup<NEIGHFLAG,DeviceType>::value> dup_f,
  SpecialVal special,
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cutsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_ljsq, 
  typename ArrayTypes<DeviceType>::t_ffloat_2d d_cut_coulsq, 
  typename Kokkos::DualView<params_lj_coul**, Kokkos::LayoutRight,DeviceType>::t_dev_const_um params,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, int eflag_either, 
  float qqrd2e, float cut_ljsq, float cut_lj_innersq, float denom_lj, float tabinnersq,
  float cut_lj3, float cut_lj6, float cut_lj3inv, float cut_lj6inv, float cut_lj_inner3inv, float cut_lj_inner6inv, float denom_lj6, float denom_lj12,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_rtable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_drtable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ftable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dftable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_ctable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_dctable,
  typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_etable, typename ArrayTypes<DeviceType>::t_ffloat_1d_randomread d_detable) {

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const float xtmp = x_float(i,0);
    const float ytmp = x_float(i,1);
    const float ztmp = x_float(i,2);
    const int itype = type(i);
    const float qtmp = q(i);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),&d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      const float factor_lj = special.lj[j >> SBBITS & 3];
      const float factor_coul = special.coul[j >> SBBITS & 3];
      j &= NEIGHMASK;
      const float delx = xtmp - x_float(j,0);
      const float dely = ytmp - x_float(j,1);
      const float delz = ztmp - x_float(j,2);
      const int jtype = type(j);
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < ((float)d_cutsq(itype,jtype))) {

        float fpair = 0.0f;

        if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (((float)params(itype,jtype).lj1)*r6inv -
            ((float)params(itype,jtype).lj2));

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
            const float table = ((float)d_ftable[itable]) + fraction*((float)d_dftable[itable]);
            float forcecoul = qtmp* ((float)q[j]) * table;
            if (factor_coul < 1.0f) {
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

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        a_f(j,0) -= delx*fpair;
        a_f(j,1) -= dely*fpair;
        a_f(j,2) -= delz*fpair;

        if (EVFLAG) {
          // F_FLOAT evdwl = 0.0;
          if (eflag_either) {
            if (rsq < ((float)d_cut_ljsq(itype,jtype))) {
              // evdwl = factor_lj * c.template compute_evdwl<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype);
              // ev.evdwl += 1.0*evdwl;
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = ((float)params(itype,jtype).lj3)*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -((float)params(itype,jtype).lj4)*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv * ((float)params(itype,jtype).lj3) * r6inv -
                ((float)params(itype,jtype).lj3) * cut_lj_inner6inv*cut_lj6inv;
                englj6 = -((float)params(itype,jtype).lj4)*r6inv +
                  ((float)params(itype,jtype).lj4)*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += 1.0f * factor_lj * englj;
            }
            if (rsq < ((float)d_cut_coulsq(itype,jtype))) {
              float ecoul = 0.0f;
              // ecoul = c.template compute_ecoul<STACKPARAMS,Specialisation>(rsq,i,j,itype,jtype,factor_coul,qtmp);
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - ((float)d_rtable[itable])) * ((float)d_drtable[itable]);
                const float table = ((float)d_etable[itable]) + fraction * ((float)d_detable[itable]);
                ecoul = qtmp * ((float)q[j]) * table;
                if (factor_coul < 1.0f) {
                  const float table = ((float)d_ctable[itable]) + fraction * ((float)d_dctable[itable]);
                  const float prefactor = qtmp * ((float)q[j]) * table;
                  ecoul -= (1.0f - factor_coul) * prefactor;
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

          // if (c.vflag_either || c.eflag_atom) ev_tally(ev,i,j,evdwl+ecoul,fpair,delx,dely,delz);
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


// template specialization for USE_SEP_SPECIAL=1 case
template<class DeviceType, int NEIGHFLAG, int EVFLAG, int EFLAG_EITHER, int REORDER_NEIGH>
__global__ void float_force_kernel_performance(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cut_sq, float cut_ljsq, float cut_coulsq,
  int ntypes, float2 *param_lj12, float2 *param_lj34,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  float qqrd2e, float cut_lj_innersq, float denom_lj, float tabinnersq,
  float cut_lj3, float cut_lj6, float cut_lj3inv, float cut_lj6inv, float cut_lj_inner3inv, float cut_lj_inner6inv, float denom_lj6, float denom_lj12,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  float* d_rtable_f, float* d_drtable_f,
  float* d_ftable_f, float* d_dftable_f,
  float* d_ctable_f, float* d_dctable_f,
  float* d_etable_f, float* d_detable_f) {

    // USE_SEP_SPECIAL = 1

    __shared__ float special_lj_shared[4];
    __shared__ float special_coul_shared[4];
    
    if (threadIdx.x < 4) {
      special_lj_shared[threadIdx.x] = (float)(special.lj[threadIdx.x]);
      special_coul_shared[threadIdx.x] = (float)(special.coul[threadIdx.x]);
    }
    __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    // auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const int neighbor_index = REORDER_NEIGH ? ii : i;
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];
    const int itype = x_data_i.type;
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(neighbor_index,0),d_numneigh(i),&d_neighbors(neighbor_index,1)-&d_neighbors(neighbor_index,0));
    const int jnum = d_numneigh(i);

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    for (int jj = 0; jj < jnum; jj++) {
      int j = neighbors_i(jj);
      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P_f * grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        // atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        // atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        // atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));
        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(neighbor_index,0),d_numneigh_special(i),&d_neighbors_special(neighbor_index,1)-&d_neighbors_special(neighbor_index,0));
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
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            if (factor_coul < 1.0) {
              const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
              const float prefactor = qtmp* qj * table;
              forcecoul -= (1.0f-factor_coul)*prefactor;
            }
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P*grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0f) forcecoul -= (1.0f-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        // atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        // atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        // atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));
        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += factor_lj * englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
                  const float prefactor = qtmp* qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P*grij);
                const float erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f-factor_coul)*prefactor;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    if (EVFLAG && EFLAG_EITHER) {
      ev_array(i) = ev;
    }
    // atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    // atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    // atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}


//Specialisation for Neighborlist types Half, HalfThread, Full
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

  PairComputeFunctorCustomFloat(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomFloat() {c.copymode = 1; list.copymode = 1;};

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

  template<int EVFLAG>
  void do_launch(int ntotal, PairStyle* fpair) {
    int threadsPerBlock = 128;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    // if (fpair -> reorder_neighbor == ON_NEIGH_BUILD) {
    //   printf("ERROR: reorder_neighbor not supported for float kernels\n");
    //   exit(1);
    // }

    auto neighbor_index = (fpair->reorder_neighbor == ON_NEIGH_BUILD) ? list.d_neigh_index : list.d_ilist;

#define LAUNCH_FLOAT_FORCE_KERNEL_PERFORMANCE(EFLAG_EITHER, REORDER_NEIGH)  \
  do {  \
    float_force_kernel_performance<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, EFLAG_EITHER, REORDER_NEIGH><<<blocksPerGrid, threadsPerBlock>>>(  \
        ntotal, neighbor_index, list.d_numneigh, list.d_neighbors, list.d_numneigh_special, list.d_neighbors_special,   \
        c.x_floatq, c.f_float,  \
        SpecialVal(c.special_coul, c.special_lj), \
        (float)c.cut_sq, (float)c.cut_ljsq, (float)c.cut_coulsq,  \
        c.atom->ntypes, c.param_lj12_f, c.param_lj34_f,   \
        c.ev_array, \
        (float)c.qqrd2e, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
        (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,  \
        c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
        c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);  \
  } while(0)


    if (fpair -> use_sep_sepcial) {
      if (c.eflag_either) {
        if (fpair->reorder_neighbor == ON_NEIGH_BUILD) {
          LAUNCH_FLOAT_FORCE_KERNEL_PERFORMANCE(1, 1);
        }
        else {
          LAUNCH_FLOAT_FORCE_KERNEL_PERFORMANCE(1, 0);
        }
      }
      else {
        if (fpair->reorder_neighbor == ON_NEIGH_BUILD) {
          LAUNCH_FLOAT_FORCE_KERNEL_PERFORMANCE(0, 1);
        }
        else {
          LAUNCH_FLOAT_FORCE_KERNEL_PERFORMANCE(0, 0);
        }
      }
    }
    else {
      float_force_kernel<typename PairStyle::device_type, NEIGHFLAG, EVFLAG><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, dup_f,
          SpecialVal(c.special_coul, c.special_lj),
          c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
          c.ev_array, c.eflag_either, 
          (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
          (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,
          c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,
          c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);        
    }
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    // printf("in kernel_launch\n");    
    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
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

    if (fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_floatq\n");
      if (fpair -> x_floatq_size > 0) {
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
      temp.type = (short)(curr_type(i));
      temp.q = (short)(curr_q(i) * Q_FACTOR);
      curr_x_floatq[i] = temp;
    
      curr_f_float[i * 3 + 0] = 0.0f;
      curr_f_float[i * 3 + 1] = 0.0f;
      curr_f_float[i * 3 + 2] = 0.0f;
    });
    Kokkos::fence();

    do_launch<0>(ntotal, fpair);

    // int threadsPerBlock = 128;
    // int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;
    
    // float_force_kernel<typename PairStyle::device_type, NEIGHFLAG, 0><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, dup_f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag_either, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    cudaDeviceSynchronize();

    curr_f_float = c.f_float;
    double* f_ptr = f.data();

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {    
      f_ptr[i * 3 + 0] += (double)(curr_f_float[i * 3 + 0]);
      f_ptr[i * 3 + 1] += (double)(curr_f_float[i * 3 + 1]);
      f_ptr[i * 3 + 2] += (double)(curr_f_float[i * 3 + 2]);
    });
    Kokkos::fence();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    // printf("ntotal value : %d\n", ntotal);
    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
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

    if (fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_floatq\n");
      if (fpair -> x_floatq_size > 0) {
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
      temp.type = (short)(curr_type(i));
      temp.q = (short)(curr_q(i) * Q_FACTOR);
      curr_x_floatq[i] = temp;
    
      curr_f_float[i * 3 + 0] = 0.0f;
      curr_f_float[i * 3 + 1] = 0.0f;
      curr_f_float[i * 3 + 2] = 0.0f;
    });
    Kokkos::fence();

    do_launch<1>(ntotal, fpair);

    // int threadsPerBlock = 128;
    // int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;
    
    // float_force_kernel<typename PairStyle::device_type, NEIGHFLAG, 1><<<blocksPerGrid, threadsPerBlock>>>(
    //     ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_float, c.type, c.q, dup_f,
    //     SpecialVal(c.special_coul, c.special_lj),
    //     c.d_cutsq, c.d_cut_ljsq, c.d_cut_coulsq, c.params, 
    //     c.ev_array, c.eflag_either, 
    //     (float)c.qqrd2e, (float)c.cut_ljsq, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq,
    //     (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,
    //     c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,
    //     c.d_rtable, c.d_drtable, c.d_ftable, c.d_dftable, c.d_ctable, c.d_dctable, c.d_etable, c.d_detable);

    cudaDeviceSynchronize();

    curr_f_float = c.f_float;
    double* f_ptr = f.data();

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {    
      f_ptr[i * 3 + 0] += (double)(curr_f_float[i * 3 + 0]);
      f_ptr[i * 3 + 1] += (double)(curr_f_float[i * 3 + 1]);
      f_ptr[i * 3 + 2] += (double)(curr_f_float[i * 3 + 2]);
    });
    Kokkos::fence();

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    // // print neighbor staistic info
    // static int curr_iter = 0;
    // if (curr_iter == 0) {
    //   FILE* file = fopen("neigh_size.txt", "w");
    //   // int* temp_neigh = new int[(fpair -> x).extent(0)];
    //   for (int i = 0; i < ntotal; i++) {
    //     fprintf(file, "%d\n", list.d_numneigh[i]);
    //   }
    //   fclose(file);
    // }
    // curr_iter++;

    // int global_max_neighbor = 0;
    // int global_min_neighbor = 1e9;
    // auto curr_numneigh = list.d_numneigh;
    // int sum_neighbor = 0;

    // Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, int& t_min_neighbor, int& t_max_neighbor, int& t_sum_neighbor) {
    //   t_min_neighbor = min(t_min_neighbor, curr_numneigh[i]);
    //   t_max_neighbor = max(t_max_neighbor, curr_numneigh[i]);
    //   t_sum_neighbor += curr_numneigh[i];
    // }, Kokkos::Min<int>(global_min_neighbor), Kokkos::Max<int>(global_max_neighbor), sum_neighbor);
    // Kokkos::fence();

    // printf("max_neighbor : %d, min_neighbor : %d, avg neighbor : %d\n", 
    //       global_max_neighbor, global_min_neighbor, sum_neighbor / ntotal);

    return ev;
  }
};


// template specialization for USE_SEP_SPECIAL=1 case
template<class DeviceType, int NEIGHFLAG, int EVFLAG, int EFLAG_EITHER, int REORDER_NEIGH>
__global__ void hfmix_force_multi_neigh_list_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_outer, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_outer,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  float* max_inner_rsq, float* min_outer_rsq,
  AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cut_sq, float cut_ljsq, float cut_coulsq,
  int ntypes, float2 *param_lj12, float2 *param_lj34,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  float qqrd2e, float cut_lj_innersq, float denom_lj, float tabinnersq,
  float cut_lj3, float cut_lj6, float cut_lj3inv, float cut_lj6inv, float cut_lj_inner3inv, float cut_lj_inner6inv, float denom_lj6, float denom_lj12,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  float* d_rtable_f, float* d_drtable_f,
  float* d_ftable_f, float* d_dftable_f,
  float* d_ctable_f, float* d_dctable_f,
  float* d_etable_f, float* d_detable_f) {

    // USE_SEP_SPECIAL = 1

    __shared__ float special_lj_shared[4];
    __shared__ float special_coul_shared[4];
    
    if (threadIdx.x < 4) {
      special_lj_shared[threadIdx.x] = (float)(special.lj[threadIdx.x]);
      special_coul_shared[threadIdx.x] = (float)(special.coul[threadIdx.x]);
    }
    __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    // auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const int neighbor_index = REORDER_NEIGH ? ii : i;
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];
    const int itype = x_data_i.type;
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    max_inner_rsq[i] = 0.0;
    min_outer_rsq[i] = 1e9;


    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i_outer = 
      AtomNeighborsConst(&d_neighbors_outer(neighbor_index,0),d_numneigh_outer(i),&d_neighbors_outer(neighbor_index,1)-&d_neighbors_outer(neighbor_index,0));
    const int jnum_outer = d_numneigh_outer(i);

    for (int jj = 0; jj < jnum_outer; jj++) {
      // int j = neighbors_i_outer(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i_outer._firstneigh[(size_t) jj * neighbors_i_outer._stride];
      int j = __ldcs(neighbors_i_raw_ptr);

      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P_f * grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(neighbor_index,0),d_numneigh(i),&d_neighbors(neighbor_index,1)-&d_neighbors(neighbor_index,0));
    const int jnum = d_numneigh(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int j = __ldcs(neighbors_i_raw_ptr);

      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P_f * grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        // atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        // atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        // atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));
        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(neighbor_index,0),d_numneigh_special(i),&d_neighbors_special(neighbor_index,1)-&d_neighbors_special(neighbor_index,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      // int j = neighbors_i_special(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i_special._firstneigh[(size_t) jj * neighbors_i_special._stride];
      int j = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;

      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp - x_data_j.x[0];
      const float dely = ytmp - x_data_j.x[1];
      const float delz = ztmp - x_data_j.x[2];
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            if (factor_coul < 1.0) {
              const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
              const float prefactor = qtmp* qj * table;
              forcecoul -= (1.0f-factor_coul)*prefactor;
            }
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P*grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0f) forcecoul -= (1.0f-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += factor_lj * englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
                  const float prefactor = qtmp* qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P*grij);
                const float erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f-factor_coul)*prefactor;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    if (EVFLAG && EFLAG_EITHER) {
      ev_array(i) = ev;
    }
    // atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    // atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    // atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}


// template specialization for USE_SEP_SPECIAL=1 case
template<class DeviceType, int NEIGHFLAG, int EVFLAG, int EFLAG_EITHER, int NEIGH_REV, int USE_RELATIVE_COORD>
__global__ void hfmix_force_basic_neigh_sep_kernel(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  float* max_inner_rsq, float* min_outer_rsq,
  AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cut_sq, float cut_ljsq, float cut_coulsq,
  int ntypes, float2 *param_lj12, float2 *param_lj34,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  float qqrd2e, float cut_lj_innersq, float denom_lj, float tabinnersq,
  float cut_lj3, float cut_lj6, float cut_lj3inv, float cut_lj6inv, float cut_lj_inner3inv, float cut_lj_inner6inv, float denom_lj6, float denom_lj12,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  float binsizex, float binsizey, float binsizez,
  float* d_rtable_f, float* d_drtable_f,
  float* d_ftable_f, float* d_dftable_f,
  float* d_ctable_f, float* d_dctable_f,
  float* d_etable_f, float* d_detable_f) {

    // USE_SEP_SPECIAL = 1

    __shared__ float special_lj_shared[4];
    __shared__ float special_coul_shared[4];
    
    if (threadIdx.x < 4) {
      special_lj_shared[threadIdx.x] = (float)(special.lj[threadIdx.x]);
      special_coul_shared[threadIdx.x] = (float)(special.coul[threadIdx.x]);
    }
    __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  if (ii < ntotal) {
    // auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int i = d_ilist(ii);
    const int neighbor_index = i;
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const __half xtmp_h = __float2half(xtmp);
    const __half ytmp_h = __float2half(ytmp);
    const __half ztmp_h = __float2half(ztmp);

    const int itype = x_data_i.type;
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float curr_max_inner_rsq = 0.0;
    float curr_min_outer_rsq = 1e9;


    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),&d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);
    const int fhcut_num = fhcut_split(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      AoS_floatq x_data_j = x_floatq[j];
      float rsq, delx, dely, delz;
      if (NEIGH_REV) {
        if (jj < fhcut_num) {
          const __half dataj_x_h = __float2half(x_data_j.x[0]);
          const __half dataj_y_h = __float2half(x_data_j.x[1]);
          const __half dataj_z_h = __float2half(x_data_j.x[2]);

          delx = USE_RELATIVE_COORD ? (__half2float(xtmp_h - dataj_x_h) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
          dely = USE_RELATIVE_COORD ? (__half2float(ytmp_h - dataj_y_h) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
          delz = USE_RELATIVE_COORD ? (__half2float(ztmp_h - dataj_z_h) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);

          rsq = delx*delx + dely*dely + delz*delz;
        }
        else {
          delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
          dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
          delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
          // const float t_delx = xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex;
          // const float t_dely = ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey;
          // const float t_delz = ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez;
          
          // const float delx = x_float(i, 0) - x_float(j, 0);
          // const float dely = x_float(i, 1) - x_float(j, 1);
          // const float delz = x_float(i, 2) - x_float(j, 2);

          // if (i == 0) {
          //   if (fabs(t_delx - delx) > 1e-5 || fabs(t_dely - dely) > 1e-5 || fabs(t_delz - delz) > 1e-5) {
          //     printf("Error coord: %d %d %f %f %f %f %f %f\n", i, j, t_delx, t_dely, t_delz, delx, dely, delz);
          //     printf("bin size : %f %f %f\n", binsizex, binsizey, binsizez);
          //     printf("dir mask : %d %d %d\n", (ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT, (ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT, (ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT);
          //     printf("relative coord : (%f %f %f), (%f %f %f)\n", xtmp, ytmp, ztmp, x_data_j.x[0], x_data_j.x[1], x_data_j.x[2]);
          //     printf("abs coord : (%f %f %f), (%f %f %f)\n", x_float(i, 0), x_float(i, 1), x_float(i, 2), x_float(j, 0), x_float(j, 1), x_float(j, 2));
          //   }  
          // }
          rsq = delx*delx + dely*dely + delz*delz;
        }
      }
      else {
        if (jj >= fhcut_num) {
          const __half dataj_x_h = __float2half(x_data_j.x[0]);
          const __half dataj_y_h = __float2half(x_data_j.x[1]);
          const __half dataj_z_h = __float2half(x_data_j.x[2]);

          delx = USE_RELATIVE_COORD ? (__half2float(xtmp_h - dataj_x_h) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
          dely = USE_RELATIVE_COORD ? (__half2float(ytmp_h - dataj_y_h) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
          delz = USE_RELATIVE_COORD ? (__half2float(ztmp_h - dataj_z_h) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);

          rsq = delx*delx + dely*dely + delz*delz;
        }
        else {
          delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
          dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
          delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);

          rsq = delx*delx + dely*dely + delz*delz;
        }
      }

      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;

      if (NEIGH_REV) {
        if (jj >= fhcut_num) {
          curr_max_inner_rsq = max(curr_max_inner_rsq, rsq);
        }
        else {
          curr_min_outer_rsq = min(curr_min_outer_rsq, rsq);
        }
      }
      else {
        if (jj < fhcut_num) {
          curr_max_inner_rsq = max(curr_max_inner_rsq, rsq);
        }
        else {
          curr_min_outer_rsq = min(curr_min_outer_rsq, rsq);
        }
      }

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P_f * grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        // atomicAdd(&f_ptr[j * 3 + 0], (double)(-delx*fpair));
        // atomicAdd(&f_ptr[j * 3 + 1], (double)(-dely*fpair));
        // atomicAdd(&f_ptr[j * 3 + 2], (double)(-delz*fpair));
        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    max_inner_rsq[i] = curr_max_inner_rsq;
    min_outer_rsq[i] = curr_min_outer_rsq;

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      // int j = neighbors_i_special(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i_special._firstneigh[(size_t) jj * neighbors_i_special._stride];
      int j = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;

      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            if (factor_coul < 1.0) {
              const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
              const float prefactor = qtmp* qj * table;
              forcecoul -= (1.0f-factor_coul)*prefactor;
            }
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P*grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0f) forcecoul -= (1.0f-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += factor_lj * englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
                  const float prefactor = qtmp* qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P*grij);
                const float erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f-factor_coul)*prefactor;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    if (EVFLAG && EFLAG_EITHER) {
      ev_array(i) = ev;
    }
    // atomicAdd(&f_ptr[i * 3 + 0], (double)fxtmp);
    // atomicAdd(&f_ptr[i * 3 + 1], (double)fytmp);
    // atomicAdd(&f_ptr[i * 3 + 2], (double)fztmp);
    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}


template<class DeviceType, int NEIGHFLAG, int EVFLAG, int EFLAG_EITHER, int NEIGH_REV, int USE_RELATIVE_COORD>
__global__ void hfmix_force_basic_neigh_sep_kernel_AoShalf(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  // float* max_inner_rsq, float* min_outer_rsq,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cut_sq, float cut_ljsq, float cut_coulsq,
  int ntypes, float2 *param_lj12, float2 *param_lj34,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  float qqrd2e, float cut_lj_innersq, float denom_lj, float tabinnersq,
  float cut_lj3, float cut_lj6, float cut_lj3inv, float cut_lj6inv, float cut_lj_inner3inv, float cut_lj_inner6inv, float denom_lj6, float denom_lj12,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  float binsizex, float binsizey, float binsizez,
  float* d_rtable_f, float* d_drtable_f,
  float* d_ftable_f, float* d_dftable_f,
  float* d_ctable_f, float* d_dctable_f,
  float* d_etable_f, float* d_detable_f) {

    // USE_SEP_SPECIAL = 1

    __shared__ float q_val_arr_shared[Q_NTYPES];
    __shared__ float special_lj_shared[4];
    __shared__ float special_coul_shared[4];
    
    if (threadIdx.x < 4) {
      special_lj_shared[threadIdx.x] = (float)(special.lj[threadIdx.x]);
      special_coul_shared[threadIdx.x] = (float)(special.coul[threadIdx.x]);
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

  if (!NEIGH_REV) {
    // set fhcut_num to the max fhcut_split value 
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
      fhcut_num = max(fhcut_num, other);
    }
  }
  if (ii < ntotal) {
    // auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int neighbor_index = i;
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = x_data_i.type;
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    // float curr_max_inner_rsq = 0.0;
    // float curr_min_outer_rsq = 1e9;

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),&d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);
    // const int fhcut_num = fhcut_split(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (NEIGH_REV) {
        if (jj < fhcut_num) {
          const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
          AoS_halfq x_data_j_h;
          uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
          *target_ptr = aligned_ptr[0];
          // AoS_floatq x_data_j = x_floatq[j];

          delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
          dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
          delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

          // delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
          // dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
          // delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
          
          rsq = delx*delx + dely*dely + delz*delz;
          jtype = x_data_j_h.type;
          // jtype = x_data_j.type;
          qj = q_val_arr_shared[x_data_j_h.q_type];
          // qj = ((float)x_data_j.q) / Q_FACTOR;
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
      }
      else {
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
      }

      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);

      // if (NEIGH_REV) {
      //   if (jj >= fhcut_num) {
      //     curr_max_inner_rsq = max(curr_max_inner_rsq, rsq);
      //   }
      //   else {
      //     curr_min_outer_rsq = min(curr_min_outer_rsq, rsq);
      //   }
      // }
      // else {
      //   if (jj < fhcut_num) {
      //     curr_max_inner_rsq = max(curr_max_inner_rsq, rsq);
      //   }
      //   else {
      //     curr_min_outer_rsq = min(curr_min_outer_rsq, rsq);
      //   }
      // }

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P_f * grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    // max_inner_rsq[i] = curr_max_inner_rsq;
    // min_outer_rsq[i] = curr_min_outer_rsq;

    // do not use relative coord for special
    const float xtmp_special = x_float(i, 0);
    const float ytmp_special = x_float(i, 1);
    const float ztmp_special = x_float(i, 2);

    const AtomNeighborsConst neighbors_i_special = 
      AtomNeighborsConst(&d_neighbors_special(i,0),d_numneigh_special(i),&d_neighbors_special(i,1)-&d_neighbors_special(i,0));
    const int jnum_special = d_numneigh_special(i);

    for (int jj = 0; jj < jnum_special; jj++) {
      // int j = neighbors_i_special(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i_special._firstneigh[(size_t) jj * neighbors_i_special._stride];
      int j = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;

      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            if (factor_coul < 1.0) {
              const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
              const float prefactor = qtmp* qj * table;
              forcecoul -= (1.0f-factor_coul)*prefactor;
            }
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P*grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0f) forcecoul -= (1.0f-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += factor_lj * englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
                  const float prefactor = qtmp* qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P*grij);
                const float erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f-factor_coul)*prefactor;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    if (EVFLAG && EFLAG_EITHER) {
      ev_array(i) = ev;
    }
    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}


template<class DeviceType, int NEIGHFLAG, int EVFLAG, int EFLAG_EITHER, int NEIGH_REV, int USE_RELATIVE_COORD>
__global__ void hfmix_force_basic_neigh_sep_kernel_AoShalf_coulTable(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace> x_float, 
  typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_special, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors_special,
  int q_val_num, float* q_val_arr,
  AoS_halfq* x_halfq, AoS_floatq* x_floatq,
  float* f_float, SpecialVal special,
  float cut_sq, float cut_ljsq, float cut_coulsq,
  int ntypes, float2 *param_lj12, float2 *param_lj34,
  Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> ev_array, 
  float qqrd2e, float cut_lj_innersq, float denom_lj, float tabinnersq,
  float cut_lj3, float cut_lj6, float cut_lj3inv, float cut_lj6inv, float cut_lj_inner3inv, float cut_lj_inner6inv, float denom_lj6, float denom_lj12,
  int ncoulmask, int ncoulshiftbits, float g_ewald,
  float binsizex, float binsizey, float binsizez, float* coul_ftable_f,
  float* d_rtable_f, float* d_drtable_f,
  float* d_ftable_f, float* d_dftable_f,
  float* d_ctable_f, float* d_dctable_f,
  float* d_etable_f, float* d_detable_f) {

    // USE_SEP_SPECIAL = 1

    __shared__ float q_val_arr_shared[Q_NTYPES];
    __shared__ float special_lj_shared[4];
    __shared__ float special_coul_shared[4];
    // __shared__ float coul_ftable_shared[COUL_RBND - COUL_LBND + 1];
    // __shared__ __half coul_ftable_shared[((COUL_RBND + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS) + 1];
    
    if (threadIdx.x < 4) {
      special_lj_shared[threadIdx.x] = (float)(special.lj[threadIdx.x]);
      special_coul_shared[threadIdx.x] = (float)(special.coul[threadIdx.x]);
    }

    for(int i = threadIdx.x; i < q_val_num; i += blockDim.x) {
      q_val_arr_shared[i] = q_val_arr[i];
    }

    // for(int i = threadIdx.x + (COUL_LBND >> COUL_BITS); i <= ((COUL_RBND + COUL_OFFSET) >> COUL_BITS); i += blockDim.x) {
    //   coul_ftable_shared[i - (COUL_LBND >> COUL_BITS)] = coul_ftable_f[i << COUL_BITS];
    // }

    __syncthreads();

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  int i = -1, fhcut_num = 0;
  if (ii < ntotal) {
    i = d_ilist(ii);
    fhcut_num = fhcut_split(i);
  }
  
  if (!NEIGH_REV) {
    // set fhcut_num to the max fhcut_split value 
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
      int other = __shfl_down_sync(0xffffffff, fhcut_num, offset);
      fhcut_num = max(fhcut_num, other);
    }
  }

  if (ii < ntotal) {
    // auto a_f = dup_f.template access<typename AtomicDup<NEIGHFLAG,DeviceType>::value>();

    EV_FLOAT ev;
    const int neighbor_index = i;
    AoS_floatq x_data_i = x_floatq[i];
    const float xtmp = x_data_i.x[0];
    const float ytmp = x_data_i.x[1];
    const float ztmp = x_data_i.x[2];

    const int itype = x_data_i.type;
    const float qtmp = ((float)x_data_i.q) / Q_FACTOR;
    // double* f_ptr = f.data();

    float fxtmp = 0.0f;
    float fytmp = 0.0f;
    float fztmp = 0.0f;

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0),d_numneigh(i),&d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);
    // const int fhcut_num = fhcut_split(i);

    for (int jj = 0; jj < jnum; jj++) {
      // int j = neighbors_i(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i._firstneigh[(size_t) jj * neighbors_i._stride];
      int ni = __ldcs(neighbors_i_raw_ptr);
      int j = ni & DIRNEIGHMASK;

      int jtype;
      float rsq, delx, dely, delz, qj;
      if (NEIGH_REV) {
        if (jj < fhcut_num) {
          const uint64_t* aligned_ptr = reinterpret_cast<const uint64_t*>(&x_halfq[j]);
          AoS_halfq x_data_j_h;
          uint64_t* target_ptr = reinterpret_cast<uint64_t*>(&x_data_j_h);
          *target_ptr = aligned_ptr[0];
          // AoS_floatq x_data_j = x_floatq[j];

          delx = USE_RELATIVE_COORD ? (xtmp - __half2float(x_data_j_h.x[0]) - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - __half2float(x_data_j_h.x[0]));
          dely = USE_RELATIVE_COORD ? (ytmp - __half2float(x_data_j_h.x[1]) - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - __half2float(x_data_j_h.x[1]));
          delz = USE_RELATIVE_COORD ? (ztmp - __half2float(x_data_j_h.x[2]) - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - __half2float(x_data_j_h.x[2]));

          // delx = USE_RELATIVE_COORD ? (xtmp - x_data_j.x[0] - ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex) : (xtmp - x_data_j.x[0]);
          // dely = USE_RELATIVE_COORD ? (ytmp - x_data_j.x[1] - ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey) : (ytmp - x_data_j.x[1]);
          // delz = USE_RELATIVE_COORD ? (ztmp - x_data_j.x[2] - ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez) : (ztmp - x_data_j.x[2]);
          
          rsq = delx*delx + dely*dely + delz*delz;
          jtype = x_data_j_h.type;
          // jtype = x_data_j.type;
          qj = q_val_arr_shared[x_data_j_h.q_type];
          // qj = ((float)x_data_j.q) / Q_FACTOR;
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
      }
      else {
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
      }

      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0f*rsq - 3.0f*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            // union_int_float_t rsq_lookup;
            // rsq_lookup.f = rsq;
            // const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            // const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            // const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            // float forcecoul = qtmp* qj * table;
            // fpair += forcecoul/rsq;

            ushort itable = __half_as_ushort(__float2half(rsq));
            // float ftable = coul_ftable_shared[itable - COUL_LBND];
            // float ftable = __half2float(coul_ftable_shared[itable - COUL_LBND]);
            // float ftable = __half2float(coul_ftable_shared[((itable + COUL_OFFSET) >> COUL_BITS) - (COUL_LBND >> COUL_BITS)]);
            float ftable = coul_ftable_f[itable];
            fpair += qtmp * qj * ftable;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P_f * grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F_f * grij * expm2);

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P_f * grij);
                const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
              }
              ev.ecoul += ecoul;
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
      // int j = neighbors_i_special(jj);
      const int* neighbors_i_raw_ptr = &neighbors_i_special._firstneigh[(size_t) jj * neighbors_i_special._stride];
      int j = __ldcs(neighbors_i_raw_ptr);

      const float factor_lj = special_lj_shared[j >> SBBITS & 3];
      const float factor_coul = special_coul_shared[j >> SBBITS & 3];
      j &= NEIGHMASK;

      AoS_floatq x_data_j = x_floatq[j];
      const float delx = xtmp_special - x_float(j, 0);
      const float dely = ytmp_special - x_float(j, 1);
      const float delz = ztmp_special - x_float(j, 2);
      const int jtype = x_data_j.type;
      const int combine_type_index = (itype - 1) * ntypes + (jtype - 1);
      const float qj = ((float)x_data_j.q) / Q_FACTOR;
      const float rsq = delx*delx + dely*dely + delz*delz;

      if (rsq < cut_sq) {

        float fpair = 0.0f;

        if (rsq < cut_ljsq) {
          const float r2inv = 1.0f/rsq;
          const float r6inv = r2inv*r2inv*r2inv;
          float forcelj, switch1;

          forcelj = r6inv *
            (param_lj12[combine_type_index].x*r6inv -
            param_lj12[combine_type_index].y);

          if (rsq > cut_lj_innersq) {
            switch1 = (cut_ljsq-rsq) * (cut_ljsq-rsq) *
                      (cut_ljsq + 2.0*rsq - 3.0*cut_lj_innersq) / denom_lj;
            forcelj = forcelj*switch1;
          }

          fpair+=factor_lj*forcelj*r2inv;
        }
        if (rsq < cut_coulsq) {
          if (rsq > tabinnersq) {
            union_int_float_t rsq_lookup;
            rsq_lookup.f = rsq;
            const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
            const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
            const float table = (float)d_ftable_f[itable] + fraction*(float)d_dftable_f[itable];
            float forcecoul = qtmp* qj * table;
            if (factor_coul < 1.0) {
              const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
              const float prefactor = qtmp* qj * table;
              forcecoul -= (1.0f-factor_coul)*prefactor;
            }
            fpair += forcecoul/rsq;
          } else {
            const float r = sqrtf(rsq);
            const float grij = g_ewald * r;
            const float expm2 = expf(-grij*grij);
            const float t = 1.0f / (1.0f + EWALD_P*grij);
            const float rinv = 1.0f/r;
            const float erfc = t * (A1_f + t * (A2_f + t * (A3_f + t * (A4_f + t * A5_f)))) * expm2;
            const float prefactor = qqrd2e * qtmp* qj *rinv;
            float forcecoul = prefactor * (erfc + EWALD_F*grij*expm2);
            if (factor_coul < 1.0f) forcecoul -= (1.0f-factor_coul)*prefactor;

            fpair += forcecoul*rinv*rinv;
          }
        }

        fxtmp += delx*fpair;
        fytmp += dely*fpair;
        fztmp += delz*fpair;

        atomicAdd(&f_float[j * 3 + 0], -delx*fpair);
        atomicAdd(&f_float[j * 3 + 1], -dely*fpair);
        atomicAdd(&f_float[j * 3 + 2], -delz*fpair);

        if (EVFLAG) {
          if (EFLAG_EITHER) {
            if (rsq < cut_ljsq) {
              const float r2inv = 1.0f/rsq;
              const float r6inv = r2inv*r2inv*r2inv;
              const float r = sqrtf(rsq);
              const float rinv = 1.0f/r;
              const float r3inv = rinv*rinv*rinv;
              float englj, englj12, englj6;

              if (rsq > cut_lj_innersq) {
                englj12 = param_lj34[combine_type_index].x*cut_lj6*
                  denom_lj12 * (r6inv - cut_lj6inv)*(r6inv - cut_lj6inv);
                englj6 = -param_lj34[combine_type_index].y*
                  cut_lj3*denom_lj6 * (r3inv - cut_lj3inv)*(r3inv - cut_lj3inv);
                englj = englj12 + englj6;
              } else {
                englj12 = r6inv*param_lj34[combine_type_index].x*r6inv -
                param_lj34[combine_type_index].x*cut_lj_inner6inv*cut_lj6inv;
                englj6 = -param_lj34[combine_type_index].y*r6inv +
                  param_lj34[combine_type_index].y*cut_lj_inner3inv*cut_lj3inv;
                englj = englj12 + englj6;
              }
              ev.evdwl += factor_lj * englj;
            }
            if (rsq < cut_coulsq) {
              float ecoul = 0.0f;
              if (rsq > tabinnersq) {
                union_int_float_t rsq_lookup;
                rsq_lookup.f = rsq;
                const int itable = (rsq_lookup.i & ncoulmask) >> ncoulshiftbits;
                const float fraction = (rsq_lookup.f - (float)d_rtable_f[itable]) * (float)d_drtable_f[itable];
                const float table = (float)d_etable_f[itable] + fraction*(float)d_detable_f[itable];
                ecoul = qtmp* qj * table;
                if (factor_coul < 1.0f) {
                  const float table = (float)d_ctable_f[itable] + fraction*(float)d_dctable_f[itable];
                  const float prefactor = qtmp* qj * table;
                  ecoul -= (1.0f-factor_coul)*prefactor;
                }
              } else {
                const float r = sqrtf(rsq);
                const float grij = g_ewald * r;
                const float expm2 = expf(-grij*grij);
                const float t = 1.0f / (1.0f + EWALD_P*grij);
                const float erfc = t * (A1+t*(A2+t*(A3+t*(A4+t*A5)))) * expm2;
                const float prefactor = qqrd2e * qtmp* qj /r;
                ecoul = prefactor * erfc;
                if (factor_coul < 1.0f) ecoul -= (1.0f-factor_coul)*prefactor;
              }
              ev.ecoul += ecoul;
            }
          }
        }
      }
    }

    if (EVFLAG && EFLAG_EITHER) {
      ev_array(i) = ev;
    }
    atomicAdd(&f_float[i * 3 + 0], fxtmp);
    atomicAdd(&f_float[i * 3 + 1], fytmp);
    atomicAdd(&f_float[i * 3 + 2], fztmp);
  }
}

//Specialisation for Neighborlist types Half, HalfThread, Full
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

  PairComputeFunctorCustomHfmix(PairStyle* c_ptr,
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
  ~PairComputeFunctorCustomHfmix() {c.copymode = 1; list.copymode = 1;};

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

  void kernel_init(int ntotal, PairStyle* fpair) {
    // require use_sep_sepcial for hfmix kernels
    if (!fpair -> use_sep_sepcial) {
      printf("ERROR: require use_sep_sepcial for hfmix kernels\n");
      exit(1);
    }

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
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

    if (fpair -> x_floatq_size < (fpair -> x).extent(0)) {
      printf("lazy init x_floatq\n");
      if (fpair -> x_floatq_size > 0) {
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
    
    if (fpair -> rsq_array_size < (fpair -> x).extent(0)) {
      fpair -> rsq_array_size = (fpair -> x).extent(0);
      cudaMalloc((void**)&(fpair -> max_inner_rsq), (fpair -> x).extent(0) * sizeof(float));
      cudaMalloc((void**)&(fpair -> min_outer_rsq), (fpair -> x).extent(0) * sizeof(float));
      c.max_inner_rsq = fpair -> max_inner_rsq;
      c.min_outer_rsq = fpair -> min_outer_rsq;
    }

    if (!fpair -> q_val_inited) {
      fpair -> q_val_inited = true;
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
    }

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

    // Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {
    //   float q_val = curr_q(i);
    //   int val_scaled = (int)(q_val * Q_FACTOR);
    //   if (curr_q_val_idx_mask[val_scaled + Q_FACTOR] == -1) {
    //     // if (Kokkos::atomic_fetch_add(&curr_q_val_idx_mask[val_scaled + Q_FACTOR], 1) == -1) {
    //     //   int index = Kokkos::atomic_fetch_add(&curr_q_val_num(), 1);
    //       curr_q_val_idx_mask[val_scaled + Q_FACTOR] = 1;
    //       curr_q_val_idx_map[val_scaled + Q_FACTOR] = 1;
    //       curr_q_val_arr[1] = 1;
    //     // }
    //   }
    // });
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
      temp.type = (short)(curr_type(i));
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
    int threadsPerBlock = 512;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    auto neighbor_index = (fpair->reorder_neighbor == ON_NEIGH_BUILD) ? list.d_neigh_index : list.d_ilist;

#define LAUNCH_HFMIX_FORCE_MULTI_NEIGH_LIST_KERNEL(EFLAG_EITHER, REORDER_NEIGH)  \
  do {  \
    hfmix_force_multi_neigh_list_kernel<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, EFLAG_EITHER, REORDER_NEIGH><<<blocksPerGrid, threadsPerBlock>>>( \
        ntotal, neighbor_index, list.d_numneigh, list.d_neighbors,  \
        list.d_numneigh_outer, list.d_neighbors_outer,  \
        list.d_numneigh_special, list.d_neighbors_special,  \
        c.max_inner_rsq, c.min_outer_rsq, \
        c.x_floatq, c.f_float,  \
        SpecialVal(c.special_coul, c.special_lj), \
        (float)c.cut_sq, (float)c.cut_ljsq, (float)c.cut_coulsq,  \
        c.atom->ntypes, c.param_lj12_f, c.param_lj34_f,   \
        c.ev_array, \
        (float)c.qqrd2e, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
        (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,  \
        c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
        c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);  \
  } while(0)

#define LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_BASIC(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)  \
  do {  \
    hfmix_force_basic_neigh_sep_kernel<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
        ntotal, neighbor_index, list.d_numneigh, list.d_neighbors,  \
        c.x_float, c.fhcut_split, \
        list.d_numneigh_special, list.d_neighbors_special,  \
        c.max_inner_rsq, c.min_outer_rsq, \
        c.x_floatq, c.f_float,  \
        SpecialVal(c.special_coul, c.special_lj), \
        (float)c.cut_sq, (float)c.cut_ljsq, (float)c.cut_coulsq,  \
        c.atom->ntypes, c.param_lj12_f, c.param_lj34_f,   \
        c.ev_array, \
        (float)c.qqrd2e, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
        (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,  \
        c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
        (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
        c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);  \
  } while(0)

#define LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)  \
  do {  \
    hfmix_force_basic_neigh_sep_kernel_AoShalf<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
        ntotal, neighbor_index, list.d_numneigh, list.d_neighbors,  \
        c.x_float, c.fhcut_split, \
        list.d_numneigh_special, list.d_neighbors_special,  \
        c.q_val_num.h_view(0), c.q_val_arr, \
        c.x_halfq, c.x_floatq, c.f_float,  \
        SpecialVal(c.special_coul, c.special_lj), \
        (float)c.cut_sq, (float)c.cut_ljsq, (float)c.cut_coulsq,  \
        c.atom->ntypes, c.param_lj12_f, c.param_lj34_f,   \
        c.ev_array, \
        (float)c.qqrd2e, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
        (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,  \
        c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
        (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, \
        c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);  \
  } while(0)


#define LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF_COULTABLE(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)  \
  do {  \
    hfmix_force_basic_neigh_sep_kernel_AoShalf_coulTable<typename PairStyle::device_type, NEIGHFLAG, EVFLAG, EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD><<<blocksPerGrid, threadsPerBlock>>>( \
        ntotal, neighbor_index, list.d_numneigh, list.d_neighbors,  \
        c.x_float, c.fhcut_split, \
        list.d_numneigh_special, list.d_neighbors_special,  \
        c.q_val_num.h_view(0), c.q_val_arr, \
        c.x_halfq, c.x_floatq, c.f_float,  \
        SpecialVal(c.special_coul, c.special_lj), \
        (float)c.cut_sq, (float)c.cut_ljsq, (float)c.cut_coulsq,  \
        c.atom->ntypes, c.param_lj12_f, c.param_lj34_f,   \
        c.ev_array, \
        (float)c.qqrd2e, (float)c.cut_lj_innersq, (float)c.denom_lj, (float)c.tabinnersq, \
        (float)c.cut_lj3, (float)c.cut_lj6, (float)c.cut_lj3inv, (float)c.cut_lj6inv, (float)c.cut_lj_inner3inv, (float)c.cut_lj_inner6inv, (float)c.denom_lj6, (float)c.denom_lj12,  \
        c.ncoulmask, c.ncoulshiftbits, (float)c.g_ewald,  \
        (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.coul_ftable_f, \
        c.d_rtable_f, c.d_drtable_f, c.d_ftable_f, c.d_dftable_f, c.d_ctable_f, c.d_dctable_f, c.d_etable_f, c.d_detable_f);  \
  } while(0)

// #define LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)  LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_BASIC(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)
// #define LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)  LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)
// #define LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)  LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF_COULTABLE(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)

#define LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD)  \
  do { \
    if (fpair -> method_type == 0) { \
      LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD); \
    } \
    else if (fpair -> method_type == 1) { \
      LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF_COULTABLE(EFLAG_EITHER, NEIGH_REV, USE_RELATIVE_COORD); \
    } \
    else { \
      printf("ERROR: unknown method type for kernel launch\n"); \
      exit(1); \
    } \
  } while(0)

    if (fpair -> neigh_sep_strategy == MULTI_NEIGH_LIST) {
      if (fpair -> use_relative_coord) {
        printf("ERROR: use_relative_coord not supported for hfmix kernels\n");
        exit(1);
      }
      if (c.eflag_either) {
        if (fpair -> reorder_neighbor == ON_NEIGH_BUILD) {
          LAUNCH_HFMIX_FORCE_MULTI_NEIGH_LIST_KERNEL(1, 1);
        }
        else {
          LAUNCH_HFMIX_FORCE_MULTI_NEIGH_LIST_KERNEL(1, 0);
        }
      }
      else {
        if (fpair -> reorder_neighbor == ON_NEIGH_BUILD) {
          LAUNCH_HFMIX_FORCE_MULTI_NEIGH_LIST_KERNEL(0, 1);
        }
        else {
          LAUNCH_HFMIX_FORCE_MULTI_NEIGH_LIST_KERNEL(0, 0);
        }
      }
    }
    else if (fpair -> neigh_sep_strategy == BASIC_NEIGH_SEP || fpair -> neigh_sep_strategy == BASIC_NEIGH_SEP_REV ||
      fpair -> neigh_sep_strategy == BASIC_NEIGH_SEP_OPT) {
      if (fpair -> reorder_neighbor == ON_NEIGH_BUILD) {
        printf("ERROR: reorder neighbor not supported for hfmix kernel basic_neigh_sep strategy\n");
        exit(1);        
      }
      if (fpair -> neigh_sep_strategy == BASIC_NEIGH_SEP_REV) {
        if (fpair -> use_relative_coord) {
          if (c.eflag_either) {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(1, 1, 1);
          }
          else {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(0, 1, 1);
          }
        }
        else {
          if (c.eflag_either) {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(1, 1, 0);
          }
          else {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(0, 1, 0);
          }
        }
      }
      else {
        // if (fpair -> use_relative_coord) {
        //   printf("ERROR: use_relative_coord not supported for hfmix kernels\n");
        //   exit(1);
        // }
        if (fpair -> use_relative_coord) {
          if (c.eflag_either) {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(1, 0, 1);
          }
          else {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(0, 0, 1);
          }
        }
        else {
          if (c.eflag_either) {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(1, 0, 0);
          }
          else {
            LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL(0, 0, 0);
          }
        }
      }
    }
    else {
      printf("ERROR: unsupported neigh_sep_strategy for hfmix kernels\n");
      exit(1);
    }
  
#undef LAUNCH_HFMIX_FORCE_MULTI_NEIGH_LIST_KERNEL
#undef LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_BASIC
#undef LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF
#undef LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL_AOSHALF_COULTABLE
#undef LAUNCH_HFMIX_FORCE_BASIC_NEIGH_SEP_KERNEL
  }

  void kernel_finalize(int ntotal, PairStyle* fpair) {
    auto curr_f_float = c.f_float;
    double* f_ptr = f.data();
    bool curr_use_relative_coord = fpair->use_relative_coord;

    Kokkos::parallel_for((c.atom)->nmax, KOKKOS_LAMBDA (const int i) {    
      f_ptr[i * 3 + 0] += (double)(curr_f_float[i * 3 + 0]);
      f_ptr[i * 3 + 1] += (double)(curr_f_float[i * 3 + 1]);
      f_ptr[i * 3 + 2] += (double)(curr_f_float[i * 3 + 2]);
    });
    Kokkos::fence();
  }

  void kernel_launch(int ntotal, PairStyle* fpair) {
    // printf("in kernel_launch\n");    

    kernel_init(ntotal, fpair);

    do_launch<0>(ntotal, fpair);
    cudaDeviceSynchronize();

    kernel_finalize(ntotal, fpair);
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    // printf("ntotal value : %d\n", ntotal);

    kernel_init(ntotal, fpair);

    do_launch<1>(ntotal, fpair);
    cudaDeviceSynchronize();

    kernel_finalize(ntotal, fpair);

    EV_FLOAT ev;
    Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace> curr_ev_array = c.ev_array;
    Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, EV_FLOAT &local_ev) {
      local_ev += curr_ev_array(i);
    }, ev);
    Kokkos::fence();

    // static int curr_iter = 0;
    // if (curr_iter == 0) {
    //   FILE* file = fopen("full_neigh_size.txt", "w");
    //   int* curr_numneigh = new int[ntotal];
    //   for(int i = 0; i < ntotal; i++) {
    //     curr_numneigh[i] = list.d_numneigh[i];
    //   }
    //   for(int i = 0; i < ntotal; i++) {

    //     const AtomNeighborsConst neighbors_i = 
    //       AtomNeighborsConst(&list.d_neighbors(i,0),list.d_numneigh(i),&list.d_neighbors(i,1)-&list.d_neighbors(i,0));

    //     for(int j = 0; j < list.d_numneigh[i]; j++) {
    //       int t = neighbors_i(j) & DIRNEIGHMASK;
    //       curr_numneigh[t]++;
    //     }
    //   }
      
    //   // int* temp_neigh = new int[(fpair -> x).extent(0)];
    //   for (int i = 0; i < ntotal; i++) {
    //     // int ii = list.d_neigh_index[i];
    //     fprintf(file, "%d %d\n", curr_numneigh[i], c.type[i]);
    //   }
    //   fclose(file);
    // }
    // curr_iter++;

    // // print neighbor and rsq staistic info
    // float* curr_max_inner_rsq = c.max_inner_rsq;
    // float* curr_min_outer_rsq = c.min_outer_rsq;
    // float global_max_inner_rsq = 0.0;
    // float global_min_outer_rsq = 1e9;
    // int global_max_neighbor = 0;
    // int global_min_neighbor = 1e9;
    // // int global_max_neighbor_outer = 0;
    // // int global_min_neighbor_outer = 1e9;
    // int global_max_neighbor_fhcut = 0;
    // int global_min_neighbor_fhcut = 1e9;
    // auto curr_numneigh = list.d_numneigh;
    // // auto curr_numneigh_outer = list.d_numneigh_outer;
    // auto curr_fhcut_split = c.fhcut_split;
    // auto curr_numneigh_special = list.d_numneigh_special;
    // // int sum_neighbor_outer = 0;
    // int sum_neighbor_fhcut = 0;

    // Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, float& t_min_outer_rsq, float& t_max_inner_rsq, int& t_min_neighbor, int& t_max_neighbor, 
    //   int& t_min_neighbor_fhcut, int& t_max_neighbor_fhcut, int& t_sum_neighbor_fhcut) {
    //   t_min_outer_rsq = min(t_min_outer_rsq, curr_min_outer_rsq[i]);
    //   t_max_inner_rsq = max(t_max_inner_rsq, curr_max_inner_rsq[i]);
    //   t_min_neighbor = min(t_min_neighbor, curr_numneigh[i]);
    //   t_max_neighbor = max(t_max_neighbor, curr_numneigh[i]);
    //   t_min_neighbor_fhcut = min(t_min_neighbor_fhcut, curr_fhcut_split[i]);
    //   t_max_neighbor_fhcut = max(t_max_neighbor_fhcut, curr_fhcut_split[i]);
    //   t_sum_neighbor_fhcut += curr_fhcut_split[i];
    // }, Kokkos::Min<float>(global_min_outer_rsq), Kokkos::Max<float>(global_max_inner_rsq), Kokkos::Min<int>(global_min_neighbor), Kokkos::Max<int>(global_max_neighbor), 
    //    Kokkos::Min<int>(global_min_neighbor_fhcut), Kokkos::Max<int>(global_max_neighbor_fhcut), sum_neighbor_fhcut);
    // Kokkos::fence();

    // printf("in kernel_launch_reduce, max_inner_rsq : %f, min_outer_rsq : %f\n", global_max_inner_rsq, global_min_outer_rsq);
    // printf("max_neighbor : %d, min_neighbor : %d, max_neighbor_fhcut : %d, min_neighbor_fhcut : %d, avg neighbor_fhcut : %d\n", 
    //       global_max_neighbor, global_min_neighbor, global_max_neighbor_fhcut, global_min_neighbor_fhcut, sum_neighbor_fhcut / ntotal);
    // fflush(stdout);

    return ev;
  }
};

template<class PairStyle, PRECISION_TYPE PRECTYPE, unsigned NEIGHFLAG, int ZEROFLAG = 0, class Specialisation = void>
EV_FLOAT pair_compute_neighlist_custom (PairStyle* fpair, typename std::enable_if<(NEIGHFLAG&PairStyle::EnabledNeighFlags) != 0, NeighListKokkos<typename PairStyle::device_type>*>::type list) {

  if(NEIGHFLAG != HALFTHREAD) {
    printf("ERROR: NEIGHFLAG is not HALFTHREAD\n");
    exit(1);
  }
  if(ZEROFLAG != 0) {
    printf("ERROR: ZEROFLAG is not 0\n");
    exit(1);
  }

  if (fpair->atom->ntypes <= MAX_TYPES_STACKPARAMS) {
    printf("ERROR: atom->ntypes is lesser than MAX_TYPES_STACKPARAMS\n");
    exit(1);
  }

  if (!fpair->newton_pair) {
    printf("ERROR: newton pair should be set to on\n");
    exit(1);
  }
  
  if (!std::is_same<typename DoCoul<PairStyle::COUL_FLAG>::type, CoulTag>::value) {
    printf("ERROR: DoCoul<PairStyle::COUL_FLAG>::type is not CoulTag\n");
    exit(1);
  }

  if (!Specialisation::DoTable) {
    printf("ERROR: Specialisation::DoTable not set\n");
    exit(1);
  }

  if (fpair->vflag_either || fpair->eflag_atom) {
    printf("ERROR: vflag_either or eflag_atom is set\n");
    exit(1);
  }

  // if (std::is_same<typename DoCoul<PairStyle::COUL_FLAG>::type, CoulTag>::value) {
  //   printf("DoCoul type is CoulTag\n");
  // }
  // else if (std::is_same<typename DoCoul<PairStyle::COUL_FLAG>::type, NoCoulTag>::value) {
  //   printf("DoCoul type is NoCoulTag\n");
  // }
  // else {
  //   printf("DoCoul type unknown\n");
  // }

  // printf("eflag : %d, vflag : %d, eflag_either : %d, vflag_either : %d, eflag_atom : %d, vflag_global : %d, vflag_atom : %d\n", 
  //   fpair->eflag, fpair->vflag, fpair->eflag_either, fpair->vflag_either, fpair->eflag_atom, fpair->vflag_global, fpair->vflag_atom);
  // printf("Specialisation::DoTable : %d\n", Specialisation::DoTable);

  EV_FLOAT ev;

  if (PRECTYPE == DOUBLE_PREC) {
    // printf("in CharmmfswCoulLongKernels::pair_compute_neighlist_custom: DOUBLE_PREC\n");
    PairComputeFunctorCustomDouble<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list);
    if (fpair->eflag || fpair->vflag) {
      // Kokkos::parallel_reduce(list->inum,ff,ev);
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      // Kokkos::parallel_for(list->inum,ff);
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
  }
  else if (PRECTYPE == FLOAT_PREC) {
    // printf("in CharmmfswCoulLongKernels::pair_compute_neighlist_custom: FLOAT_PREC\n");
    PairComputeFunctorCustomFloat<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
    ff.contribute();
  }
  else if (PRECTYPE == HFMIX_PREC) {
    // printf("in CharmmfswCoulLongKernels::pair_compute_neighlist_custom: HFMIX_PREC\n");
    PairComputeFunctorCustomHfmix<PairStyle,NEIGHFLAG,false,ZEROFLAG,Specialisation > ff(fpair,list);
    if (fpair->eflag || fpair->vflag) {
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

} // namespace CharmmfswCoulLongKernels
