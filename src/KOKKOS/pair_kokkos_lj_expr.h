namespace LJKernelsExpr {

template <class DeviceType, int EVFLAG>
__global__ void double_optimized(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
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

template <class PairStyle, int NEIGHFLAG, bool STACKPARAMS, int USE_RELATIVE_COORD, NEIGH_SEP_STRATEGY NEIGH_STG = NO_NEIGH_SEP, int ZEROFLAG = 0, class Specialisation = void>
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

  NeighListKokkos<device_type> list;

  PairComputeFunctorCustomDouble(PairStyle* c_ptr,
                          NeighListKokkos<device_type>* list_ptr, int ntotal):
  c(*c_ptr),list(*list_ptr) {
    // allocate duplicated memory
    f = c.f;
    d_eatom = c.d_eatom;
    d_vatom = c.d_vatom;
    Kokkos::Timer init_timer;
    init_timer.reset();
    //ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
    c_ptr->init_time += init_timer.seconds();
  };

  // Set copymode = 1 so parent allocations aren't destructed by copies of the style
  ~PairComputeFunctorCustomDouble() {c.copymode = 1; list.copymode = 1;};

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n");

    if(!fpair -> ev_allocated) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    //int threadsPerBlock = 128;
    int threadsPerBlock = 512;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    double_optimized<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
        ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, f,
        c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

    cudaDeviceSynchronize();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", ntotal);
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    //int threadsPerBlock = 128;
    int threadsPerBlock = 512;
    int blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    double_optimized<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
        ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x, c.type, f, 
        c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);

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

  void kernel_launch(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // MemoryKokkos::realloc_kokkos
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }
    if (USE_RELATIVE_COORD) {
      if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
        // printf("lazy init x_float_rel\n"); fflush(stdout);
        fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float_rel = fpair -> x_float_rel;
        printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
      }
    }
    else {
      if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)) {
        // printf("lazy init x_float\n"); fflush(stdout);
        fpair -> x_float = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float = fpair -> x_float;
        // printf("x_float extent : %d, %d\n", (fpair -> x_float).extent(0), (fpair -> x_float).extent(1));
      }
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

    //int threadsPerBlock = 128;
    int threadsPerBlock = 512;
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
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if (USE_RELATIVE_COORD) {
      if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
        // printf("lazy init x_float_rel\n"); fflush(stdout);
        fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
        fpair -> x_float_allocated = true;
        c.x_float_rel = fpair -> x_float_rel;
        printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
      }
    }
    else {
      if(!fpair -> x_float_allocated || (fpair -> x_float).extent(0) < (fpair -> x).extent(0)){
        // printf("lazy init x_float\n");
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

    //int threadsPerBlock = 128;
    int threadsPerBlock = 512;
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

      __half2 delx_h2, dely_h2, delz_h2;
      delx_h2.x = __float2half(xtmp_f - __half2float(half_data_j1.x[0]) - ((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      delx_h2.y = __float2half(xtmp_f - __half2float(half_data_j2.x[0]) - ((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      dely_h2.x = __float2half(ytmp_f - __half2float(half_data_j1.x[1]) - ((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      dely_h2.y = __float2half(ytmp_f - __half2float(half_data_j2.x[1]) - ((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      delz_h2.x = __float2half(ztmp_f - __half2float(half_data_j1.x[2]) - ((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);
      delz_h2.y = __float2half(ztmp_f - __half2float(half_data_j2.x[2]) - ((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);

    //   const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
    //     - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
    //                      __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
    //   const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
    //     - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
    //                      __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
    //   const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
    //     - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
    //                      __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

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

      __half2 delx_h2, dely_h2, delz_h2;
      delx_h2.x = __float2half(xtmp_f - __half2float(half_data_j1.x[0]) - ((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      delx_h2.y = __float2half(xtmp_f - __half2float(half_data_j2.x[0]) - ((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      dely_h2.x = __float2half(ytmp_f - __half2float(half_data_j1.x[1]) - ((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      dely_h2.y = __float2half(ytmp_f - __half2float(half_data_j2.x[1]) - ((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      delz_h2.x = __float2half(ztmp_f - __half2float(half_data_j1.x[2]) - ((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);
      delz_h2.y = __float2half(ztmp_f - __half2float(half_data_j2.x[2]) - ((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);

    //   const __half2 delx_h2 = xtmp_h2 - __halves2half2(half_data_j1.x[0], half_data_j2.x[0]) 
    //     - __halves2half2(__int2half_rn((ni1 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT), 
    //                      __int2half_rn((ni2 & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizex));
    //   const __half2 dely_h2 = ytmp_h2 - __halves2half2(half_data_j1.x[1], half_data_j2.x[1]) 
    //     - __halves2half2(__int2half_rn((ni1 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT), 
    //                      __int2half_rn((ni2 & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizey));
    //   const __half2 delz_h2 = ztmp_h2 - __halves2half2(half_data_j1.x[2], half_data_j2.x[2]) 
    //     - __halves2half2(__int2half_rn((ni1 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT), 
    //                      __int2half_rn((ni2 & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT)) * __half2half2(__float2half(binsizez));

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
__global__ void __launch_bounds__(BLOCK_SIZE, 4) fhmix_expr_optimized_drift(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d d_ilist,
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

      // __half2 delx_h2, dely_h2, delz_h2;
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

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n"); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // MemoryKokkos::realloc_kokkos
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
      // printf("lazy init x_float_rel\n"); fflush(stdout);
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

    //printf("float kernel launch part0\n"); fflush(stdout);

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    //printf("float kernel launch part1\n"); fflush(stdout);
    
    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    init_aos_xfhmix_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
        nmax, c.x_half_rel_xonly, c.x_float_rel, c.type, c.x, c.bin_base);

    cudaDeviceSynchronize();

    // threadsPerBlock = 128;
    threadsPerBlock = BLOCK_SIZE;
    // threadsPerBlock = 512;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      if (NEIGH_STG == TWO_END_NEIGH) {
        fhmix_two_end_half2_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
            ntotal, list.d_ilist, list.d_numfront, list.d_numback, list.d_neighbors, list.maxneighs, c.x_half_rel_xonly, c.x_float_rel,
            (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
            c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      }
      else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
        if (fpair -> method_type == 1) {
          fhmix_expr_optimized_drift<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
              ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
              (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
              c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
        }
        else {
          fhmix_expr_optimized<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
              ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
              (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
              c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
        }
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
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    // printf("launch cuda reduce kernel, ntotal : %d, f.extent(0) : %d\n", ntotal, f.extent(0)); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
      // printf("lazy init x_float_rel\n"); fflush(stdout);
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

    if(!std::is_same<typename PairStyle::device_type, Kokkos::Cuda>::value) {
      printf("ERROR: device_type is not Cuda\n");
      exit(1);
    }

    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    init_aos_xfhmix_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
        nmax, c.x_half_rel_xonly, c.x_float_rel, c.type, c.x, c.bin_base);

    cudaDeviceSynchronize();

    // threadsPerBlock = 128;
    threadsPerBlock = BLOCK_SIZE;
    // threadsPerBlock = 512;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      if (NEIGH_STG == TWO_END_NEIGH) {
      fhmix_two_end_half2_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numfront, list.d_numback, list.d_neighbors, list.maxneighs, c.x_half_rel_xonly, c.x_float_rel,
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
      }
      else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
        if (fpair -> method_type == 1) {
          fhmix_expr_optimized_drift<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
              ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
              (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
              c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
        }
        else {
          fhmix_expr_optimized<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
              ntotal, list.d_ilist, list.d_numneigh, list.d_numneigh_int2, list.d_neighbors, list.d_neighbors_int2, c.x_half_rel_xonly, c.x_float_rel,
              (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, c.type, f,
              c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
        }
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
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh,
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, AoS_half* x_half,
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

    __half zero_h = __ushort_as_half((unsigned short)0x0000U);
    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    __half fxtmp = zero_h;
    __half fytmp = zero_h;
    __half fztmp = zero_h;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj ++) {
      int j = neighbors_i(jj);
      const __half factor_lj = one_h;
      j &= NEIGHMASK;
      const AoS_half half_data_j = x_half[j];

      const __half delx = xtmp - half_data_j.x[0];
      const __half dely = ytmp - half_data_j.x[1];
      const __half delz = ztmp - half_data_j.x[2];

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
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh,
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors, 
  AoS_half* x_half_rel, const float binsizex, const float binsizey, const float binsizez,
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

    __half zero_h = __ushort_as_half((unsigned short)0x0000U);
    __half one_h = __ushort_as_half((unsigned short)0x3C00U);

    const AtomNeighborsConst neighbors_i = 
      AtomNeighborsConst(&d_neighbors(i,0), d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
    const int jnum = d_numneigh(i);

    __half fxtmp = zero_h;
    __half fytmp = zero_h;
    __half fztmp = zero_h;

    f(i,0) = 0.0;
    f(i,1) = 0.0;
    f(i,2) = 0.0;

    for (int jj = 0; jj < jnum; jj ++) {
      int ni = neighbors_i(jj);
      const __half factor_lj = one_h;
      int j = ni & DIRNEIGHMASK;
      const AoS_half half_data_j = x_half_rel[j];

      const __half delx = xtmp - __float2half(__half2float(half_data_j.x[0]) + ((ni & (int)DIRXMASK) << DIRLEFTSHIFTX >> DIRRIGHTSHIFT) * binsizex);
      const __half dely = ytmp - __float2half(__half2float(half_data_j.x[1]) + ((ni & (int)DIRYMASK) << DIRLEFTSHIFTY >> DIRRIGHTSHIFT) * binsizey);
      const __half delz = ztmp - __float2half(__half2float(half_data_j.x[2]) + ((ni & (int)DIRZMASK) << DIRLEFTSHIFTZ >> DIRRIGHTSHIFT) * binsizez);

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

// half precision kernel
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

  void kernel_launch(int ntotal, PairStyle* fpair) {
    //printf("launch cuda kernel\n"); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // MemoryKokkos::realloc_kokkos
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
      // printf("lazy init x_float_rel\n"); fflush(stdout);
      fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
      fpair -> x_float_allocated = true;
      c.x_float_rel = fpair -> x_float_rel;
      printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
    }
    // if(fpair -> x_half_size < (fpair -> x).extent(0)) {
    //   printf("lazy init x_half_rel_xonly\n"); fflush(stdout);
    //   if(fpair -> x_half_size) {
    //     cudaFree(fpair -> x_half_rel_xonly);
    //   }
    //   cudaMalloc((void**)&(fpair -> x_half_rel_xonly), (fpair -> x).extent(0) * sizeof(AoS_half_xonly));
    //   fpair -> x_half_size = (fpair -> x).extent(0);
    //   c.x_half_rel_xonly = fpair -> x_half_rel_xonly;
    //   printf("x_half_rel_xonly extent : %d\n", fpair -> x_half_size);
    // }
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

    //printf("float kernel launch part1\n"); fflush(stdout);
    
    int nmax = (c.atom)->nmax;
    int threadsPerBlock = 128;
    int blocksPerGrid = (nmax + threadsPerBlock - 1) / threadsPerBlock;

    // init_aos_xfhmix_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
    //     nmax, c.x_half_rel_xonly, c.x_float_rel, c.type, c.x, c.bin_base);
    if (USE_RELATIVE_COORD) {
      init_aos_xhalf_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel, c.type, c.x, c.bin_base);
    }
    else {
      init_aos_xhalf_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half, c.type, c.x);
    }

    cudaDeviceSynchronize();

    // threadsPerBlock = 128;
    threadsPerBlock = BLOCK_SIZE;
    // threadsPerBlock = 512;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      half_force_kernel_x_rel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel,
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }
    else {
      half_force_kernel<typename PairStyle::device_type, 0><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
    }
    cudaDeviceSynchronize();

    fpair -> cuda_kernel_time += cuda_kernel_timer.seconds();
  }

  EV_FLOAT kernel_launch_reduce(int ntotal, PairStyle* fpair) {
    // printf("launch cuda reduce kernel, ntotal : %d, f.extent(0) : %d\n", ntotal, f.extent(0)); fflush(stdout);

    if(!fpair -> ev_allocated || (fpair -> ev_array).extent(0) < f.extent(0)) {
      // printf("lazy init ev_array\n");
      fpair -> ev_array = Kokkos::View<EV_FLOAT*, Kokkos::CudaSpace>("ev_array", f.extent(0));
      fpair -> ev_allocated = true;
      c.ev_array = fpair -> ev_array;
    }

    if(!fpair -> x_float_allocated || (fpair -> x_float_rel).extent(0) < (fpair -> x).extent(0)) {
      // printf("lazy init x_float_rel\n"); fflush(stdout);
      fpair -> x_float_rel = Kokkos::View<float**, Kokkos::LayoutRight, Kokkos::CudaSpace>("x_float_rel", (fpair -> x).extent(0), 3);
      fpair -> x_float_allocated = true;
      c.x_float_rel = fpair -> x_float_rel;
      printf("x_float_rel extent : %d, %d\n", (fpair -> x_float_rel).extent(0), (fpair -> x_float_rel).extent(1));
    }
    // if(fpair -> x_half_size < (fpair -> x).extent(0)) {
    //   printf("lazy init x_half_rel_xonly\n"); fflush(stdout);
    //   if(fpair -> x_half_size) {
    //     cudaFree(fpair -> x_half_rel_xonly);
    //   }
    //   cudaMalloc((void**)&(fpair -> x_half_rel_xonly), (fpair -> x).extent(0) * sizeof(AoS_half_xonly));
    //   fpair -> x_half_size = (fpair -> x).extent(0);
    //   c.x_half_rel_xonly = fpair -> x_half_rel_xonly;
    //   printf("x_half_rel_xonly extent : %d\n", fpair -> x_half_size);
    // }
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

    // init_aos_xfhmix_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
    //     nmax, c.x_half_rel_xonly, c.x_float_rel, c.type, c.x, c.bin_base);
    if (USE_RELATIVE_COORD) {
      init_aos_xhalf_rel_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half_rel, c.type, c.x, c.bin_base);
    }
    else {
      init_aos_xhalf_kernel<typename PairStyle::device_type><<<blocksPerGrid, threadsPerBlock>>>(
          nmax, c.x_half, c.type, c.x);
    }

    cudaDeviceSynchronize();

    // threadsPerBlock = 128;
    threadsPerBlock = BLOCK_SIZE;
    // threadsPerBlock = 512;
    blocksPerGrid = (ntotal + threadsPerBlock - 1) / threadsPerBlock;

    Kokkos::Timer cuda_kernel_timer;
    cuda_kernel_timer.reset();

    if (USE_RELATIVE_COORD) {
      half_force_kernel_x_rel<typename PairStyle::device_type, 1><<<blocksPerGrid, threadsPerBlock>>>(
          ntotal, list.d_ilist, list.d_numneigh, list.d_neighbors, c.x_half_rel,
          (float)c.binsizex, (float)c.binsizey, (float)c.binsizez, f,
          c.ev_array, c.eflag, c.vflag_either, c.vflag_global, (float)c.m_cutsq[1][1]);
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
    PairComputeFunctorCustomDouble<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD, NEIGH_STG, ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
  }
  else if(PRECTYPE == FLOAT_PREC) {
    PairComputeFunctorCustomFloat<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD, NEIGH_STG, ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
  }
  else if(PRECTYPE == HFMIX_PREC) {
    PairComputeFunctorCustomFhmix<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD, NEIGH_STG, ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
  }
  else if(PRECTYPE == HALF_PREC) {
    PairComputeFunctorCustomHalf<PairStyle,NEIGHFLAG,true, USE_RELATIVE_COORD, NEIGH_STG, ZEROFLAG,Specialisation > ff(fpair,list,list->inum);
    if (fpair->eflag || fpair->vflag) {
      ev = ff.kernel_launch_reduce(list->inum, fpair);
    }
    else {
      ff.kernel_launch(list->inum, fpair);
    }
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


} // namespace LJKernelsExpr
