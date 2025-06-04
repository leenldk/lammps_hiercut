// clang-format off
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

#include "npair_kokkos.h"
#include "atom_kokkos.h"
#include "atom_masks.h"
#include "domain_kokkos.h"
#include "update.h"
#include "neighbor_kokkos.h"
#include "nbin_kokkos.h"
#include "nstencil.h"
#include "force.h"
#include "kokkos.h"
#include "transpose_helper_kokkos.h"

namespace LAMMPS_NS {

/* ---------------------------------------------------------------------- */

template<class DeviceType, int HALF, int NEWTON, int GHOST, int TRI, int SIZE>
NPairKokkos<DeviceType,HALF,NEWTON,GHOST,TRI,SIZE>::NPairKokkos(LAMMPS *lmp) : NPair(lmp) {

  last_stencil_old = -1;

  // use 1D view for scalars to reduce GPU memory operations

  d_scalars = typename AT::t_int_1d("neighbor:scalars",5);
  h_scalars = HAT::t_int_1d("neighbor:scalars_mirror",5);

  d_resize = Kokkos::subview(d_scalars,0);
  d_new_maxneighs = Kokkos::subview(d_scalars,1);
  d_new_maxneighs_outer = Kokkos::subview(d_scalars,2);
  d_new_maxneighs_int2 = Kokkos::subview(d_scalars,3);
  d_new_maxneighs_special = Kokkos::subview(d_scalars,4);

  h_resize = Kokkos::subview(h_scalars,0);
  h_new_maxneighs = Kokkos::subview(h_scalars,1);
  h_new_maxneighs_outer = Kokkos::subview(h_scalars,2);
  h_new_maxneighs_int2 = Kokkos::subview(h_scalars,3);
  h_new_maxneighs_special = Kokkos::subview(h_scalars,4);
}

/* ----------------------------------------------------------------------
   copy needed info from Neighbor class to this build class
   ------------------------------------------------------------------------- */

template<class DeviceType, int HALF, int NEWTON, int GHOST, int TRI, int SIZE>
void NPairKokkos<DeviceType,HALF,NEWTON,GHOST,TRI,SIZE>::copy_neighbor_info()
{
  NPair::copy_neighbor_info();

  NeighborKokkos* neighborKK = (NeighborKokkos*) neighbor;

  // general params

  k_cutneighsq = neighborKK->k_cutneighsq;

  // overwrite per-type Neighbor cutoffs with custom value set by requestor
  // only works for style = BIN (checked by Neighbor class)

  if (cutoff_custom > 0.0) {
    int n = atom->ntypes;
    auto k_mycutneighsq = DAT::tdual_xfloat_2d("neigh:cutneighsq,",n+1,n+1);
    for (int i = 1; i <= n; i++)
      for (int j = 1; j <= n; j++)
        k_mycutneighsq.h_view(i,j) = cutoff_custom * cutoff_custom;
    k_cutneighsq = k_mycutneighsq;
  }
  //printf("cutneighsq : %f, cutoff_custom : %f\n", k_cutneighsq.h_view(1,1), cutoff_custom);
  printf("cut_sq in NPairKokkos : %f\n", force->pair->cutsq[1][1]);

  k_cutneighsq.modify<LMPHostType>();

  // exclusion info

  k_ex1_type = neighborKK->k_ex1_type;
  k_ex2_type = neighborKK->k_ex2_type;
  k_ex_type = neighborKK->k_ex_type;
  k_ex1_bit = neighborKK->k_ex1_bit;
  k_ex2_bit = neighborKK->k_ex2_bit;
  k_ex_mol_group = neighborKK->k_ex_mol_group;
  k_ex_mol_bit = neighborKK->k_ex_mol_bit;
  k_ex_mol_intra = neighborKK->k_ex_mol_intra;
}

/* ----------------------------------------------------------------------
 copy per-atom and per-bin vectors from NBin class to this build class
 ------------------------------------------------------------------------- */

template<class DeviceType, int HALF, int NEWTON, int GHOST, int TRI, int SIZE>
void NPairKokkos<DeviceType,HALF,NEWTON,GHOST,TRI,SIZE>::copy_bin_info()
{
  NPair::copy_bin_info();

  NBinKokkos<DeviceType>* nbKK = (NBinKokkos<DeviceType>*) nb;

  atoms_per_bin = nbKK->atoms_per_bin;
  k_bincount = nbKK->k_bincount;
  k_bins = nbKK->k_bins;
  k_atom2bin = nbKK->k_atom2bin;
}

/* ----------------------------------------------------------------------
 copy needed info from NStencil class to this build class
 ------------------------------------------------------------------------- */

template<class DeviceType, int HALF, int NEWTON, int GHOST, int TRI, int SIZE>
void NPairKokkos<DeviceType,HALF,NEWTON,GHOST,TRI,SIZE>::copy_stencil_info()
{
  NPair::copy_stencil_info();
  nstencil = ns->nstencil;

  if (ns->last_stencil != last_stencil_old || ns->last_stencil == update->ntimestep) {
    // copy stencil to device as it may have changed

    last_stencil_old = ns->last_stencil;

    int maxstencil = ns->get_maxstencil();

    if (maxstencil > (int)k_stencil.extent(0))
      k_stencil = DAT::tdual_int_1d("neighlist:stencil",maxstencil);
    for (int k = 0; k < maxstencil; k++) {
      k_stencil.h_view(k) = ns->stencil[k];
      //printf("stencil %d : %d\n", k, ns->stencil[k]);
    }
    k_stencil.modify<LMPHostType>();
    k_stencil.sync<DeviceType>();

    k_stencil_dir = DAT::tdual_int_1d_3("neighlist:stencil_dir",maxstencil);
    if (HALF) {
      int t_nstencil = 0;
      // for (int k = 0; k <= ns->sz; k++)
      //   for (int j = -ns->sy; j <= ns->sy; j++)
      //     for (int i = -ns->sx; i <= ns->sx; i++)
      //       if (k > 0 || j > 0 || (j == 0 && i > 0))
      //         if (ns->bin_distance(i, j, k) < ns->cutneighmaxsq) {
      //           // stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;
      //           k_stencil_dir.h_view(t_nstencil, 0) = i;
      //           k_stencil_dir.h_view(t_nstencil, 1) = j;
      //           k_stencil_dir.h_view(t_nstencil, 2) = k;
      //           t_nstencil++;
      //         }
      int sy_min = ns->sy;
      int sz_min = ns->sz;
      if ((!TRI) && HALF) sz_min = 0;
    
      if (HALF && (!TRI)) {
        // stencil[nstencil++] = 0;
        k_stencil_dir.h_view(t_nstencil, 0) = 0;
        k_stencil_dir.h_view(t_nstencil, 1) = 0;
        k_stencil_dir.h_view(t_nstencil, 2) = 0;
        t_nstencil++;  
      }
      for (int k = -sz_min; k <= ns->sz; k++) {
        for (int j = -sy_min; j <= ns->sy; j++) {
          for (int i = -ns->sx; i <= ns->sx; i++) {
            if (HALF && (!TRI))
              if (k <= 0 && j <= 0 && (j != 0 || i <= 0)) continue;
    
            if (ns->bin_distance(i, j, k) < ns->cutneighmaxsq) {
              // stencil[nstencil++] = k * mbiny * mbinx + j * mbinx + i;
              k_stencil_dir.h_view(t_nstencil, 0) = i;
              k_stencil_dir.h_view(t_nstencil, 1) = j;
              k_stencil_dir.h_view(t_nstencil, 2) = k;
              t_nstencil++;
            }
          }
        }
      }      
      // printf("stencil for HALF neighbor, stencil dir : %d %d %d, stencil size : %d, nstencil : %d\n", ns->sx, ns->sy, ns->sz, t_nstencil, nstencil);
    }
    else {
      int t_nstencil = 0;
      for (int k = -ns->sz; k <= ns->sz; k++)
        for (int j = -ns->sy; j <= ns->sy; j++)
          for (int i = -ns->sx; i <= ns->sx; i++)
            if (ns->bin_distance(i, j, k) < ns->cutneighmaxsq) {
              k_stencil_dir.h_view(t_nstencil, 0) = i;
              k_stencil_dir.h_view(t_nstencil, 1) = j;
              k_stencil_dir.h_view(t_nstencil, 2) = k;
              t_nstencil++;
            }
      // printf("stencil for FULL neighbor, stencil dir : %d %d %d, stencil size : %d, nstencil : %d\n", ns->sx, ns->sy, ns->sz, t_nstencil, nstencil);
    }
    k_stencil_dir.modify<LMPHostType>();
    k_stencil_dir.sync<DeviceType>();
    if(ns->sz > 1 || ns->sy > 1 || ns->sx > 1) {
      printf("error stencil size > 1\n");
      exit(1);
    }
    
    if (GHOST) {
      if (maxstencil > (int)k_stencilxyz.extent(0))
        k_stencilxyz = DAT::tdual_int_1d_3("neighlist:stencilxyz",maxstencil);
      for (int k = 0; k < maxstencil; k++) {
        k_stencilxyz.h_view(k,0) = ns->stencilxyz[k][0];
        k_stencilxyz.h_view(k,1) = ns->stencilxyz[k][1];
        k_stencilxyz.h_view(k,2) = ns->stencilxyz[k][2];
      }
      k_stencilxyz.modify<LMPHostType>();
      k_stencilxyz.sync<DeviceType>();
    }
  }
}

/* ---------------------------------------------------------------------- */

template <class DeviceType>
__global__ void build_ReleativeX(const int nall, typename ArrayTypes<DeviceType>::t_int_1d_const c_atom2bin, 
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, typename ArrayTypes<DeviceType>::t_x_array x_rel,
  typename ArrayTypes<DeviceType>::t_x_array bin_base,
  const int mbinx, const int mbiny, const int mbinz, const X_FLOAT binsizex, const X_FLOAT binsizey, const X_FLOAT binsizez,
  const int mbinxlo, const int mbinylo, const int mbinzlo, const X_FLOAT bboxlo_x, const X_FLOAT bboxlo_y, const X_FLOAT bboxlo_z) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= nall) return;

  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  const int ibin = c_atom2bin(i);

  int ix = ibin % mbinx;
  int iy = (ibin/mbinx) % mbiny;
  int iz = ibin/(mbinx*mbiny);

  X_FLOAT basex = (ix + mbinxlo + 0.5) * binsizex + bboxlo_x;
  X_FLOAT basey = (iy + mbinylo + 0.5) * binsizey + bboxlo_y;
  X_FLOAT basez = (iz + mbinzlo + 0.5) * binsizez + bboxlo_z;

  bin_base(i, 0) = basex;
  bin_base(i, 1) = basey;
  bin_base(i, 2) = basez;
  //X_FLOAT basex, basey, basez;
  //bin_base_coord(basex, basey, basez, ibin);

  x_rel(i, 0) = xtmp - basex;
  x_rel(i, 1) = ytmp - basey;
  x_rel(i, 2) = ztmp - basez;

  // if(fabs(x_rel(i, 0)) > 30) {
  //   printf("error coord too big\n");
  //   printf("in build_ReleativeX ibin : %d, ix %d, iy %d, iz %d, basex %f, basey %f, basez %f\n", ibin, ix, iy, iz, basex, basey, basez);
  //   printf("i: %d, x: %f, y: %f, z: %f, x_rel: %f, y_rel: %f, z_rel: %f\n", i, xtmp, ytmp, ztmp, x_rel(i, 0), x_rel(i, 1), x_rel(i, 2));
  // }
  // if(i == 33708134) {
  //   printf("in build_ReleativeX ibin : %d, ix %d, iy %d, iz %d, basex %f, basey %f, basez %f\n", ibin, ix, iy, iz, basex, basey, basez);
  //   printf("i: %d, x: %f, y: %f, z: %f, x_rel: %f, y_rel: %f, z_rel: %f\n", i, xtmp, ytmp, ztmp, x_rel(i, 0), x_rel(i, 1), x_rel(i, 2));
  // }
}

namespace UtilFuncs {

constexpr int max_index_value = 2000;

template <class DeviceType>
__global__ void bucket_sort_kernel(int ntotal, int bin_size, int* bin_array, typename ArrayTypes<DeviceType>::t_int_1d index, 
  typename ArrayTypes<DeviceType>::t_int_1d index_inv, typename ArrayTypes<DeviceType>::t_int_1d value) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= ntotal || blockIdx.x > 0) return;

  for(int i = idx; i < bin_size; i += blockDim.x) {
    bin_array[i] = 0;
  }
  __syncthreads();
  
  for(int i = idx; i < ntotal; i += blockDim.x) {
    int idx = min(max_index_value, value[i]);
    atomicAdd(&(bin_array[idx]), 1);
  }
  __syncthreads();
  
  if(idx == 0) {
    for(int i = 1; i < bin_size; i++) {
      bin_array[i] += bin_array[i - 1];
    }
  }
  __syncthreads();

  for(int i = idx; i < ntotal; i += blockDim.x) {
    int idx = min(max_index_value, value[i]);
    int t = atomicAdd(&(bin_array[idx]), -1);
    index[ntotal - t] = i;
    index_inv[i] = ntotal - t;
  }  
}

// use bucket sort to sort ntotal elements according to value (ensure that elements in value >= 0)
template <class DeviceType>
void bucket_sort(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d index, 
  typename ArrayTypes<DeviceType>::t_int_1d index_inv, typename ArrayTypes<DeviceType>::t_int_1d value) {
  static int bin_array_size = 0;
  static int* bin_array = nullptr;
  
  int bin_size = 0; 
  Kokkos::parallel_reduce(ntotal, KOKKOS_LAMBDA (const int i, int& t_bin_size) {
    t_bin_size = max(t_bin_size, value[i]);
  }, Kokkos::Max<int>(bin_size));
  Kokkos::fence();
  if (bin_size > max_index_value) {
    bin_size = max_index_value;
  }
  bin_size++;
  printf("kokkos sort : set bin size to %d\n", bin_size);
  
  if (bin_array_size < bin_size) {
    bin_array_size = bin_size * 1.2;
    if (bin_array) {
      cudaFree(bin_array);
    }
    cudaMalloc((void**)&bin_array, bin_array_size * sizeof(int));
  }

  bucket_sort_kernel<DeviceType><<<1, 1024>>>(ntotal, bin_size, bin_array, index, index_inv, value);
  cudaDeviceSynchronize();
}

// init index to identity
template <class DeviceType>
void init_id_index(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d index, 
  typename ArrayTypes<DeviceType>::t_int_1d index_inv) {
  Kokkos::parallel_for(ntotal, KOKKOS_LAMBDA (const int i) {
    index[i] = i;
    index_inv[i] = i;
  });
  Kokkos::fence();  
}

};

// kernel function for use_basic_fhcut, NEIGH_REV controls whether to reverse two parts of neighbor
template <class DeviceType, int NEIGH_REV>
__global__ void neigh_sort(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, const float rsq_fhcut) {
  
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntotal) return;
  
  const AtomNeighbors neighbors_i = 
    AtomNeighbors(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
  const int num_neighs_i = d_numneigh(i);

  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  // const X_FLOAT rsq_fhcut = cutsq * p_fhcut * p_fhcut;
  //printf("%f %f\n", cutsq, rsq_fhcut);

  int head = 0; 
  int tail = num_neighs_i - 1;

  while(head <= tail) {
    while(head <= tail) {
      int idx = neighbors_i(head) & DIRNEIGHMASK;
      const X_FLOAT dx = xtmp - x(idx, 0);
      const X_FLOAT dy = ytmp - x(idx, 1);
      const X_FLOAT dz = ztmp - x(idx, 2);
      const X_FLOAT rsq = dx*dx + dy*dy + dz*dz;
      if (NEIGH_REV) {
        if (rsq <= rsq_fhcut) {
          break;
        }
      }
      else {
        if (rsq > rsq_fhcut) {
          break;
        }
      }
      head++;
    }
    while(head <= tail) {
      T_INT idx = neighbors_i(tail) & DIRNEIGHMASK;
      const X_FLOAT dx = xtmp - x(idx, 0);
      const X_FLOAT dy = ytmp - x(idx, 1);
      const X_FLOAT dz = ztmp - x(idx, 2);
      const X_FLOAT rsq = dx*dx + dy*dy + dz*dz;
      if (NEIGH_REV) {
        if (rsq > rsq_fhcut) {
          break;
        }
      }
      else {
        if (rsq <= rsq_fhcut) {
          break;
        }
      }
      tail--;
    }
    if(head < tail) {
      int t = neighbors_i(head); 
      neighbors_i(head) = neighbors_i(tail);
      neighbors_i(tail) = t;
      //swap(neighs_i(head), neighs_i(tail));
    }
  }

  // make the half range diviable by 2
  // if((head ^ num_neighs_i) & 1) {
  //   head++;
  // }

  fhcut_split(i) = head;  
}


// optimized kernel to perform neighbor sort, based on a variation of birdirectional neighbor separation strategy
// inner neighbors at the front, while outer neighbors at the back
template <class DeviceType>
__global__ void neigh_sort_opt(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numfront, typename ArrayTypes<DeviceType>::t_int_1d d_numback, int maxneighs,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors) {
  
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntotal) return;
  
  const AtomNeighbors neighbors_i = 
    AtomNeighbors(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
  int num_front = d_numfront(i);
  int num_back = d_numback(i);
  fhcut_split(i) = num_front;  
  d_numneigh(i) = num_front + (maxneighs - num_back);

  int target_num = num_front + (maxneighs - num_back);
  if (target_num > num_back) {
    int t = target_num; target_num = num_back; num_back = t;
  }

  // for(int i = num_back; i < maxneighs; i++) {
  //   neighbors_i(num_front++) = neighbors_i(i);
  // }
  for(int i = maxneighs - 1; i >= num_back; i--) {
    neighbors_i(num_front++) = neighbors_i(i);
  }
}

// adjust neighbor order based on previous fhcut
template <class DeviceType>
__global__ void neigh_sort_adjust(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numfront, typename ArrayTypes<DeviceType>::t_int_1d d_numback, int maxneighs,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors) {

  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntotal) return;

  int num_front = d_numfront(i);
  int prev_fhcut = fhcut_split(i);
  int num_back = d_numback(i);

  const AtomNeighbors neighbors_i = 
    AtomNeighbors(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
  int num_neighbor = d_numneigh(i);

  if (prev_fhcut > num_front) {

    int move_count = prev_fhcut - num_front;
    int new_num_neighbor = num_neighbor - move_count;
    int move_start = num_neighbor - move_count;

    if (move_count > num_neighbor - prev_fhcut) {
      move_count = num_neighbor - prev_fhcut;
      new_num_neighbor = num_front + move_count;
      move_start = prev_fhcut;
    }
    for(int j = 0; j < move_count; j++) {
      neighbors_i(num_front + j) = neighbors_i(move_start + j);
    }
    d_numneigh(i) = new_num_neighbor;
    fhcut_split(i) = num_front;
  }
  else if (num_back < maxneighs) {

    int move_count = maxneighs - num_back;
    int new_num_neighbor = num_neighbor + move_count;
    int move_target = num_neighbor;

    if (move_count > num_neighbor - prev_fhcut) {
      move_target = prev_fhcut + move_count;
    }
    for(int j = 0; j < move_count; j++) {
      int t = neighbors_i(prev_fhcut + j);
      neighbors_i(prev_fhcut + j) = neighbors_i(num_back + j);
      neighbors_i(move_target + j) = t;
    }
    d_numneigh(i) = new_num_neighbor;
    fhcut_split(i) = prev_fhcut + move_count;
  }
}

template <class DeviceType, int NEIGH_REV>
__global__ void verify_neigh_sort(int ntotal, typename ArrayTypes<DeviceType>::t_int_1d fhcut_split, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh, typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, const float rsq_fhcut) {
  
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntotal) return;
  
  const AtomNeighbors neighbors_i = 
    AtomNeighbors(&d_neighbors(i,0),d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
  const int num_neighs_i = d_numneigh(i);
  int fhcut_split_i = fhcut_split(i);

  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  // const X_FLOAT rsq_fhcut = cutsq * p_fhcut * p_fhcut;

  for(int j = 0; j < fhcut_split_i; j++) {
    int idx = neighbors_i(j) & DIRNEIGHMASK;
    const X_FLOAT dx = xtmp - x(idx, 0);
    const X_FLOAT dy = ytmp - x(idx, 1);
    const X_FLOAT dz = ztmp - x(idx, 2);
    const X_FLOAT rsq = dx*dx + dy*dy + dz*dz;
    if (NEIGH_REV) {
      if (rsq <= rsq_fhcut) {
        printf("error in verify_neigh_sort, i: %d, j: %d, fhcut_split: %d, idx: %d, rsq: %f, rsq_fhcut: %f\n", i, j, fhcut_split_i, idx, rsq, rsq_fhcut);
      }
    }
    else {
      if (rsq > rsq_fhcut) {
        printf("error in verify_neigh_sort, i: %d, j: %d, fhcut_split: %d, idx: %d, rsq: %f, rsq_fhcut: %f\n", i, j, fhcut_split_i, idx, rsq, rsq_fhcut);
      }
    }
  }
  for(int j = fhcut_split_i; j < num_neighs_i; j++) {
    int idx = neighbors_i(j) & DIRNEIGHMASK;
    const X_FLOAT dx = xtmp - x(idx, 0);
    const X_FLOAT dy = ytmp - x(idx, 1);
    const X_FLOAT dz = ztmp - x(idx, 2);
    const X_FLOAT rsq = dx*dx + dy*dy + dz*dz;
    if (NEIGH_REV) {
      if (rsq > rsq_fhcut) {
        printf("error in verify_neigh_sort, i: %d, j: %d, fhcut_split: %d, idx: %d, rsq: %f, rsq_fhcut: %f\n", i, j, fhcut_split_i, idx, rsq, rsq_fhcut);
      }
    }
    else {
      if (rsq <= rsq_fhcut) {
        printf("error in verify_neigh_sort, i: %d, j: %d, fhcut_split: %d, idx: %d, rsq: %f, rsq_fhcut: %f\n", i, j, fhcut_split_i, idx, rsq, rsq_fhcut);
      }
    }
  }  
}

// kernel function for use_two_end_neigh
template <class DeviceType>
__global__ void verify_two_end_neigh(int ntotal, int maxneighs, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numfront, typename ArrayTypes<DeviceType>::t_int_1d d_numback,
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x, const float rsq_fhcut) {
  
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntotal) return;
  
  const AtomNeighbors neighbors_i = 
    AtomNeighbors(&d_neighbors(i,0), maxneighs, &d_neighbors(i,1)-&d_neighbors(i,0));
  const int num_neighs_front = d_numfront(i);
  const int num_neighs_back = d_numback(i);

  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);

  // const X_FLOAT rsq_fhcut = cutsq * p_fhcut * p_fhcut;

  for(int j = 0; j < num_neighs_front; j++) {
    int idx = neighbors_i(j) & DIRNEIGHMASK;
    const X_FLOAT dx = xtmp - x(idx, 0);
    const X_FLOAT dy = ytmp - x(idx, 1);
    const X_FLOAT dz = ztmp - x(idx, 2);
    const X_FLOAT rsq = dx*dx + dy*dy + dz*dz;
    if (rsq <= rsq_fhcut) {
      printf("error in verify_two_end_neigh, i: %d, j: %d, num_neighs_front: %d, num_neighs_back %d, maxneighs: %d, idx: %d, rsq: %f, rsq_fhcut: %f\n", 
        i, j, num_neighs_front, num_neighs_back, maxneighs, idx, rsq, rsq_fhcut);
    }
  }
  for(int j = num_neighs_back; j < maxneighs; j++) {
    int idx = neighbors_i(j) & DIRNEIGHMASK;
    const X_FLOAT dx = xtmp - x(idx, 0);
    const X_FLOAT dy = ytmp - x(idx, 1);
    const X_FLOAT dz = ztmp - x(idx, 2);
    const X_FLOAT rsq = dx*dx + dy*dy + dz*dz;
    if (rsq > rsq_fhcut) {
      printf("error in verify_two_end_neigh, i: %d, j: %d, num_neighs_front: %d, num_neighs_back %d, maxneighs: %d, idx: %d, rsq: %f, rsq_fhcut: %f\n", 
        i, j, num_neighs_front, num_neighs_back, maxneighs, idx, rsq, rsq_fhcut);
    }
  }  
}

template <class DeviceType>
__global__ void neigh_exchange_int2(int ntotal, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh,
  typename ArrayTypes<DeviceType>::t_int_1d d_numneigh_int2,
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_neighbors_2d_int2 d_neighbors_int2,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x) {
  
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntotal) return;
  
  const AtomNeighbors neighbors_i = 
    AtomNeighbors(&d_neighbors(i,0), d_numneigh(i), &d_neighbors(i,1)-&d_neighbors(i,0));
  const AtomNeighbors_int2 neighbors_i_int2 = 
    AtomNeighbors_int2(&d_neighbors_int2(i,0), d_numneigh_int2(i), &d_neighbors_int2(i,1)-&d_neighbors_int2(i,0));
  const int num_neighs = d_numneigh(i);
  const int num_neighs_int2 = d_numneigh_int2(i);

  if(num_neighs_int2 & 1) {
    neighbors_i(num_neighs) = neighbors_i_int2(num_neighs_int2 >> 1).x;
    d_numneigh(i)++;
    d_numneigh_int2(i)--;
  }
}

template <class DeviceType>
__global__ void neigh_exchange(int ntotal, int maxneighs, 
  typename ArrayTypes<DeviceType>::t_int_1d d_numfront,
  typename ArrayTypes<DeviceType>::t_int_1d d_numback,
  typename ArrayTypes<DeviceType>::t_neighbors_2d d_neighbors,
  typename ArrayTypes<DeviceType>::t_x_array_randomread x) {
  
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= ntotal) return;
  
  const AtomNeighbors neighbors_i = 
    AtomNeighbors(&d_neighbors(i,0), maxneighs, &d_neighbors(i,1)-&d_neighbors(i,0));
  const int num_neighs_front = d_numfront(i);
  const int num_neighs_back = d_numback(i);

  if(num_neighs_front & 1) {
    neighbors_i(num_neighs_back - 1) = neighbors_i(num_neighs_front - 1);
    d_numfront(i)--;
    d_numback(i)--;
  }
}

template<class DeviceType, int HALF, int NEWTON, int GHOST, int TRI, int SIZE>
void NPairKokkos<DeviceType,HALF,NEWTON,GHOST,TRI,SIZE>::build(NeighList *list_)
{
  NeighListKokkos<DeviceType>* list = (NeighListKokkos<DeviceType>*) list_;
  const int nlocal = includegroup?atom->nfirst:atom->nlocal;
  int nall = nlocal;
  // printf("GHOST value : %d, nghost : %d\n", GHOST, atom->nghost);
  if (GHOST)
    nall += atom->nghost;

  if (nall == 0) return;

  if (atomKK->k_x.extent(0) != atomKK->k_x_rel.extent(0)) {
    // printf("resize x_rel to %d\n", atomKK->k_x.extent(0));
    //atomKK->k_x_rel = ArrayTypes<DeviceType>::tdual_x_array("atom:x_rel",atomKK->k_x.extent(0));
    atomKK->k_x_rel = DAT::tdual_x_array("atom:x_rel",atomKK->k_x.extent(0));
    atomKK->k_bin_base = DAT::tdual_x_array("atom:bin_base",atomKK->k_x.extent(0));
    // atomKK->k_fhcut_split_prev = DAT::tdual_int_1d("atom:k_fhcut_split_prev",atomKK->k_x.extent(0));
    atomKK->k_fhcut_split = DAT::tdual_int_1d("atom:fhcut_split",atomKK->k_x.extent(0));
    // auto fhcut_split_prev = atomKK->k_fhcut_split_prev.view<DeviceType>();
    auto curr_fhcut_split = atomKK->k_fhcut_split.view<DeviceType>();
    Kokkos::parallel_for(atomKK->k_x.extent(0), KOKKOS_LAMBDA (const int i) {    
      // fhcut_split_prev(i) = 0;
      curr_fhcut_split(i) = 0;
    });
    Kokkos::fence();
  } 
  atomKK->binsizex = nb->binsizex;
  atomKK->binsizey = nb->binsizey;
  atomKK->binsizez = nb->binsizez;

  int grow_flag = list->grow(nall);
  
  if (this->force->pair->reorder_neighbor == ON_NEIGH_BUILD) {
    if(grow_flag) {
      // neighbor list grow, thus invalidates numneigh from last iter
      UtilFuncs::init_id_index<DeviceType>(nlocal, list->d_neigh_index, list->d_neigh_index_inv);
    }
    else {
      if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
        UtilFuncs::bucket_sort<DeviceType>(nlocal, list->d_neigh_index, list->d_neigh_index_inv, list->d_numneigh_outer);
      }
      else {
        UtilFuncs::bucket_sort<DeviceType>(nlocal, list->d_neigh_index, list->d_neigh_index_inv, list->d_numneigh);
      }
    }
  }

  const double delta = 0.01 * force->angstrom;
  //printf("in NPairKokkos::build half:%d, newton:%d, ghost:%d, tri:%d, size:%d\n",
  //      HALF,NEWTON,GHOST,TRI,SIZE);

  NeighborKokkosExecute<DeviceType>
    data(*list,
         k_cutneighsq.view<DeviceType>(),
         k_bincount.view<DeviceType>(),
         k_bins.view<DeviceType>(),
         k_atom2bin.view<DeviceType>(),
         mbins,nstencil,
         k_stencil.view<DeviceType>(),
         k_stencilxyz.view<DeviceType>(),
         k_stencil_dir.view<DeviceType>(),
         nlocal,nall,lmp->kokkos->neigh_transpose,
         atomKK->k_x.view<DeviceType>(),
         atomKK->k_x_rel.view<DeviceType>(),
         atomKK->k_bin_base.view<DeviceType>(),
         atomKK->k_fhcut_split.view<DeviceType>(),
         atomKK->k_radius.view<DeviceType>(),
         atomKK->k_type.view<DeviceType>(),
         atomKK->k_mask.view<DeviceType>(),
         atomKK->k_molecule.view<DeviceType>(),
         atomKK->k_tag.view<DeviceType>(),
         atomKK->k_special.view<DeviceType>(),
         atomKK->k_nspecial.view<DeviceType>(),
         atomKK->molecular, force->pair->p_fhcut, 
         force->pair->fhcut_value * force->pair->fhcut_value,
         nbinx,nbiny,nbinz,mbinx,mbiny,mbinz,mbinxlo,mbinylo,mbinzlo,
         bininvx,bininvy,bininvz,
         nb->binsizex,nb->binsizey,nb->binsizez,
         delta, exclude, nex_type,
         k_ex1_type.view<DeviceType>(),
         k_ex2_type.view<DeviceType>(),
         k_ex_type.view<DeviceType>(),
         nex_group,
         k_ex1_bit.view<DeviceType>(),
         k_ex2_bit.view<DeviceType>(),
         nex_mol,
         k_ex_mol_group.view<DeviceType>(),
         k_ex_mol_bit.view<DeviceType>(),
         k_ex_mol_intra.view<DeviceType>(),
         bboxhi,bboxlo,
         domain->xperiodic,domain->yperiodic,domain->zperiodic,
         domain->xprd_half,domain->yprd_half,domain->zprd_half,
         skin,d_resize,h_resize,d_new_maxneighs,h_new_maxneighs,
         d_new_maxneighs_outer, h_new_maxneighs_outer,
         d_new_maxneighs_int2, h_new_maxneighs_int2,
         d_new_maxneighs_special, h_new_maxneighs_special);

  k_cutneighsq.sync<DeviceType>();
  k_ex1_type.sync<DeviceType>();
  k_ex2_type.sync<DeviceType>();
  k_ex_type.sync<DeviceType>();
  k_ex1_bit.sync<DeviceType>();
  k_ex2_bit.sync<DeviceType>();
  k_ex_mol_group.sync<DeviceType>();
  k_ex_mol_bit.sync<DeviceType>();
  k_ex_mol_intra.sync<DeviceType>();
  k_bincount.sync<DeviceType>();
  k_bins.sync<DeviceType>();
  k_atom2bin.sync<DeviceType>();

  //printf("atom->molecular : %d, Atom::ATOMIC : %d\n", atom->molecular, Atom::ATOMIC);

  if (atom->molecular != Atom::ATOMIC) {
    if (exclude)
      atomKK->sync(Device,X_MASK|RADIUS_MASK|TYPE_MASK|MASK_MASK|MOLECULE_MASK|TAG_MASK|SPECIAL_MASK);
    else
      atomKK->sync(Device,X_MASK|RADIUS_MASK|TYPE_MASK|TAG_MASK|SPECIAL_MASK);
  } else {
    if (exclude)
      atomKK->sync(Device,X_MASK|RADIUS_MASK|TYPE_MASK|MASK_MASK);
    else
      atomKK->sync(Device,X_MASK|RADIUS_MASK|TYPE_MASK);
  }

  if (HALF && NEWTON && TRI) atomKK->sync(Device,TAG_MASK);

  data.special_flag[0] = special_flag[0];
  data.special_flag[1] = special_flag[1];
  data.special_flag[2] = special_flag[2];
  data.special_flag[3] = special_flag[3];

  Kokkos::Timer neighbor_build_timer;
  neighbor_build_timer.reset();

  data.h_resize()=1;
  while(true) {
    if (!data.h_resize()) {
      break;
    }
    data.h_new_maxneighs() = list->maxneighs;
    data.h_resize() = 0;
    if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
      data.h_new_maxneighs_outer() = list->maxneighs_outer;
    }
    else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
      data.h_new_maxneighs_int2() = list->maxneighs_int2;
    }
    if (this->force->pair->use_sep_sepcial) {
      data.h_new_maxneighs_special() = list->maxneighs_special;
    }

    Kokkos::deep_copy(d_scalars, h_scalars);

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    #define BINS_PER_BLOCK 2
    const int factor = atoms_per_bin <64?2:1;
#else
    const int factor = 1;
#endif

    if (GHOST) {
      // assumes newton off

      NPairKokkosBuildFunctorGhost<DeviceType,HALF> f(data,atoms_per_bin * 5 * sizeof(X_FLOAT) * factor);

// temporarily disable team policy for ghost due to known bug

//#ifdef LMP_KOKKOS_GPU
//      if (ExecutionSpaceFromDevice<DeviceType>::space == Device) {
//        int team_size = atoms_per_bin*factor;
//        int team_size_max = Kokkos::TeamPolicy<DeviceType>(team_size,Kokkos::AUTO).team_size_max(f,Kokkos::ParallelForTag());
//        if (team_size <= team_size_max) {
//          Kokkos::TeamPolicy<DeviceType> config((mbins+factor-1)/factor,team_size);
//          Kokkos::parallel_for(config, f);
//        } else { // fall back to flat method
//          f.sharedsize = 0;
//          Kokkos::parallel_for(nall, f);
//        }
//      } else
//        Kokkos::parallel_for(nall, f);
//#else
      Kokkos::parallel_for(nall, f);
//#endif
    } else {
      if (SIZE) {
        NPairKokkosBuildFunctorSize<DeviceType,HALF,NEWTON,TRI> f(data,atoms_per_bin * 7 * sizeof(X_FLOAT) * factor);
#ifdef LMP_KOKKOS_GPU
        if (ExecutionSpaceFromDevice<DeviceType>::space == Device) {
          int team_size = atoms_per_bin*factor;
          int team_size_max = Kokkos::TeamPolicy<DeviceType>(team_size,Kokkos::AUTO).team_size_max(f,Kokkos::ParallelForTag());
          if (team_size <= team_size_max) {
            Kokkos::TeamPolicy<DeviceType> config((mbins+factor-1)/factor,team_size);
            Kokkos::parallel_for(config, f);
          } else { // fall back to flat method
            f.sharedsize = 0;
            Kokkos::parallel_for(nall, f);
          }
        } else
          Kokkos::parallel_for(nall, f);
#else
        Kokkos::parallel_for(nall, f);
#endif
      } else {
        auto launchFunctor = [&](auto functor) {
#ifdef LMP_KOKKOS_GPU
          if (ExecutionSpaceFromDevice<DeviceType>::space == Device) {
            int team_size = atoms_per_bin*factor;
            int team_size_max = Kokkos::TeamPolicy<DeviceType>(team_size,Kokkos::AUTO).team_size_max(functor,Kokkos::ParallelForTag());
            if (team_size <= team_size_max) {
              //printf("NPairKokkos::build TeamPolicy\n");
              Kokkos::TeamPolicy<DeviceType> config((mbins+factor-1)/factor,team_size);
              Kokkos::parallel_for(config, functor);
            } else { // fall back to flat method
              //printf("NPairKokkos::build flat\n");
              functor.sharedsize = 0;
              Kokkos::parallel_for(nall, functor);
            }
          } else
            Kokkos::parallel_for(nall, functor);
#else
          Kokkos::parallel_for(nall, functor);
#endif
        };
        //printf("in NPairKokkos::build NPairKokkosBuildFunctor\n"); fflush(stdout);


#define LAUNCH_NPAIR_FUNCTOR(USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH, NEIGH_STG)                                \
  do {                                                                       \
    NPairKokkosBuildFunctor<DeviceType, HALF, NEWTON, TRI, USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH, NEIGH_STG> f( \
      data, atoms_per_bin * 6 * sizeof(X_FLOAT) * factor);                   \
    launchFunctor(f);                                                        \
  } while(0)


// neigh_sep_strategy NO_NEIGH_SEP, BASIC_NEIGH_SEP and BASIC_NEIGH_SEP_REV needs no extra operation
#define LAUNCH_LOGIC_NEIGH_STG(USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH)   \
  do {    \
    if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH) {   \
      LAUNCH_NPAIR_FUNCTOR(USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH, TWO_END_NEIGH);   \
    }   \
    else if (this->force->pair->neigh_sep_strategy == BASIC_NEIGH_SEP_OPT) {   \
      LAUNCH_NPAIR_FUNCTOR(USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH, BASIC_NEIGH_SEP_OPT);   \
    }   \
    else if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {   \
      LAUNCH_NPAIR_FUNCTOR(USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH, MULTI_NEIGH_LIST);    \
    }   \
    else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {   \
      LAUNCH_NPAIR_FUNCTOR(USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH, TWO_END_NEIGH_INT2);    \
    }   \
    else {    \
      LAUNCH_NPAIR_FUNCTOR(USE_REL_COORD, USE_SEP_SPECIAL, REORDER_NEIGH, NO_NEIGH_SEP);    \
    }   \
  } while(0)


#define LAUNCH_LOGIC_REORDER_NEIGH(USE_REL_COORD, USE_SEP_SPECIAL)    \
  do {    \
    if (this->force->pair->reorder_neighbor == ON_NEIGH_BUILD) {    \
      LAUNCH_LOGIC_NEIGH_STG(USE_REL_COORD, USE_SEP_SPECIAL, 1);    \
    }   \
    else {    \
      LAUNCH_LOGIC_NEIGH_STG(USE_REL_COORD, USE_SEP_SPECIAL, 0);    \
    }   \
  } while(0)


#define LAUNCH_LOGIC_USE_SEP_SPECIAL(USE_REL_COORD)    \
  do {    \
    if (this->force->pair->use_sep_sepcial) {    \
      LAUNCH_LOGIC_REORDER_NEIGH(USE_REL_COORD, 1);    \
    }   \
    else {    \
      LAUNCH_LOGIC_REORDER_NEIGH(USE_REL_COORD, 0);    \
    }   \
  } while(0)


#define LAUNCH_LOGIC_USE_REL_COORD()    \
  do {    \
    if (this->force->pair->use_relative_coord) {    \
      LAUNCH_LOGIC_USE_SEP_SPECIAL(1);    \
    }   \
    else {    \
      LAUNCH_LOGIC_USE_SEP_SPECIAL(0);    \
    }   \
  } while(0)


        LAUNCH_LOGIC_USE_REL_COORD();
        // // template variables : use_relative_coord, neigh_sep_strategy, use_sep_sepcial
        // if (this->force->pair->use_relative_coord) {
        //   if (this->force->pair->use_sep_sepcial) {
        //     if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH) {
        //       LAUNCH_NPAIR_FUNCTOR(1, 1, TWO_END_NEIGH);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
        //       LAUNCH_NPAIR_FUNCTOR(1, 1, MULTI_NEIGH_LIST);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
        //       LAUNCH_NPAIR_FUNCTOR(1, 1, TWO_END_NEIGH_INT2);
        //     }
        //     else {
        //       // NO_NEIGH_SEP and BASIC_NEIGH_SEP needs no extra operation here
        //       LAUNCH_NPAIR_FUNCTOR(1, 1, NO_NEIGH_SEP);
        //     }
        //   }
        //   else {
        //     if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH) {
        //       LAUNCH_NPAIR_FUNCTOR(1, 0, TWO_END_NEIGH);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
        //       LAUNCH_NPAIR_FUNCTOR(1, 0, MULTI_NEIGH_LIST);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
        //       LAUNCH_NPAIR_FUNCTOR(1, 0, TWO_END_NEIGH_INT2);
        //     }
        //     else {
        //       // NO_NEIGH_SEP and BASIC_NEIGH_SEP needs no extra operation here
        //       LAUNCH_NPAIR_FUNCTOR(1, 0, NO_NEIGH_SEP);
        //     }
        //   }
        // }
        // else {
        //   if (this->force->pair->use_sep_sepcial) {
        //     if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH) {
        //       LAUNCH_NPAIR_FUNCTOR(0, 1, TWO_END_NEIGH);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
        //       LAUNCH_NPAIR_FUNCTOR(0, 1, MULTI_NEIGH_LIST);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
        //       LAUNCH_NPAIR_FUNCTOR(0, 1, TWO_END_NEIGH_INT2);
        //     }
        //     else {
        //       // NO_NEIGH_SEP and BASIC_NEIGH_SEP needs no extra operation here
        //       LAUNCH_NPAIR_FUNCTOR(0, 1, NO_NEIGH_SEP);
        //     }
        //   }
        //   else {
        //     if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH) {
        //       LAUNCH_NPAIR_FUNCTOR(0, 0, TWO_END_NEIGH);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
        //       LAUNCH_NPAIR_FUNCTOR(0, 0, MULTI_NEIGH_LIST);
        //     }
        //     else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
        //       LAUNCH_NPAIR_FUNCTOR(0, 0, TWO_END_NEIGH_INT2);
        //     }
        //     else {
        //       // NO_NEIGH_SEP and BASIC_NEIGH_SEP needs no extra operation here
        //       LAUNCH_NPAIR_FUNCTOR(0, 0, NO_NEIGH_SEP);
        //     }
        //   }
        // }

#undef LAUNCH_NPAIR_FUNCTOR
#undef LAUNCH_LOGIC_NEIGH_STG
#undef LAUNCH_LOGIC_REORDER_NEIGH
#undef LAUNCH_LOGIC_USE_SEP_SPECIAL
#undef LAUNCH_LOGIC_USE_REL_COORD

        // printf("neigh_transpose : %d\n", lmp->kokkos->neigh_transpose);
        // if (std::is_same<typename decltype(list->d_neighbors)::traits::array_layout, Kokkos::LayoutLeft>::value) {
        //   printf("d_neighbors array_layout is LayoutLeft\n");
        // } 
        // if (std::is_same<typename decltype(list->d_neighbors)::traits::array_layout, Kokkos::LayoutRight>::value) {
        //   printf("d_neighbors array_layout is LayoutRight\n");
        // }
        // int strides[2];
        // (list->d_neighbors).stride(strides);
        // printf("array list->d_neighbors stride : (%d, %d)\n", strides[0], strides[1]);
        int threadsPerBlock;
        int blocksPerGrid;

        if (this->force->pair->use_relative_coord) {
          Kokkos::fence();
          //printf("RELATIVE_COORD defined, start build_ReleativeX\n");
          //Kokkos::parallel_for("calculate_relative_x", Kokkos::RangePolicy<FunctorTags::ReleativeXTag>(0, nall), f);
          int nall_ghost = nlocal + atom->nghost;
          threadsPerBlock = 128;
          blocksPerGrid = (nall_ghost + threadsPerBlock - 1) / threadsPerBlock;
          build_ReleativeX<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(nall_ghost, data.c_atom2bin, 
            data.x, data.x_rel, data.bin_base, data.mbinx, data.mbiny, data.mbinz, data.binsizex, data.binsizey, data.binsizez, 
            data.mbinxlo, data.mbinylo, data.mbinzlo, data.bboxlo[0], data.bboxlo[1], data.bboxlo[2]);
          
          // double sum_rel_x = 0;
          // double sum_rel_y = 0;
          // double sum_rel_z = 0;
          // double max_rel_x = 0;
          // double max_rel_y = 0;
          // double max_rel_z = 0;
          // double min_rel_x = 0;
          // double min_rel_y = 0;
          // double min_rel_z = 0;
          // double range_rel_x_l = 0;
          // double range_rel_y_l = 0;
          // double range_rel_z_l = 0;
          // double range_rel_x_h = 0;
          // double range_rel_y_h = 0;
          // double range_rel_z_h = 0;

          // Kokkos::parallel_reduce(nall_ghost, KOKKOS_LAMBDA(const int i, double &lsum_rel_x, double &lsum_rel_y, double &lsum_rel_z, 
          //   double &lmax_rel_x, double &lmax_rel_y, double &lmax_rel_z,
          //   double &lmin_rel_x, double &lmin_rel_y, double &lmin_rel_z,
          //   double &lrange_rel_x_l, double &lrange_rel_y_l, double &lrange_rel_z_l,
          //   double &lrange_rel_x_h, double &lrange_rel_y_h, double &lrange_rel_z_h) {
          //   lsum_rel_x += fabs(data.x_rel(i, 0));
          //   lsum_rel_y += fabs(data.x_rel(i, 1));
          //   lsum_rel_z += fabs(data.x_rel(i, 2));
          //   lmax_rel_x = fmax(lmax_rel_x, fabs(data.x_rel(i, 0)));
          //   lmax_rel_y = fmax(lmax_rel_y, fabs(data.x_rel(i, 1)));
          //   lmax_rel_z = fmax(lmax_rel_z, fabs(data.x_rel(i, 2)));
          //   lmin_rel_x = fmin(lmin_rel_x, fabs(data.x_rel(i, 0)));
          //   lmin_rel_y = fmin(lmin_rel_y, fabs(data.x_rel(i, 1)));
          //   lmin_rel_z = fmin(lmin_rel_z, fabs(data.x_rel(i, 2)));
          //   lrange_rel_x_l = fmin(lrange_rel_x_l, data.x_rel(i, 0));
          //   lrange_rel_y_l = fmin(lrange_rel_y_l, data.x_rel(i, 1));
          //   lrange_rel_z_l = fmin(lrange_rel_z_l, data.x_rel(i, 2));
          //   lrange_rel_x_h = fmax(lrange_rel_x_h, data.x_rel(i, 0));
          //   lrange_rel_y_h = fmax(lrange_rel_y_h, data.x_rel(i, 1));
          //   lrange_rel_z_h = fmax(lrange_rel_z_h, data.x_rel(i, 2));
          // }, sum_rel_x, sum_rel_y, sum_rel_z, 
          //   Kokkos::Max<double>(max_rel_x), Kokkos::Max<double>(max_rel_y), Kokkos::Max<double>(max_rel_z),
          //   Kokkos::Min<double>(min_rel_x), Kokkos::Min<double>(min_rel_y), Kokkos::Min<double>(min_rel_z),
          //   Kokkos::Min<double>(range_rel_x_l), Kokkos::Min<double>(range_rel_y_l), Kokkos::Min<double>(range_rel_z_l),
          //   Kokkos::Max<double>(range_rel_x_h), Kokkos::Max<double>(range_rel_y_h), Kokkos::Max<double>(range_rel_z_h));
          // Kokkos::fence();

          // printf("mbinxlo : %d, mbinylo : %d, mbinzlo : %d\n", data.mbinxlo, data.mbinylo, data.mbinzlo);
          // printf("bin size : %lf, %lf, %lf\n", data.binsizex, data.binsizey, data.binsizez);
          // printf("avg rel coord : %lf, %lf, %lf, max rel coord : %lf %lf %lf, min rel coord : %lf %lf %lf\n", 
          // sum_rel_x/nall_ghost, sum_rel_y/nall_ghost, sum_rel_z/nall_ghost, max_rel_x, max_rel_y, max_rel_z, min_rel_x, min_rel_y, min_rel_z);
          // printf("range rel coord : %lf %lf %lf %lf %lf %lf\n", range_rel_x_l, range_rel_y_l, range_rel_z_l, range_rel_x_h, range_rel_y_h, range_rel_z_h);

          cudaDeviceSynchronize();
        }
      }
    }
    Kokkos::deep_copy(h_scalars, d_scalars);

    if (data.h_resize()) {
      list->maxneighs = data.h_new_maxneighs() * 1.2;
      int maxatoms = list->d_neighbors.extent(0);
      data.neigh_list.d_neighbors = typename AT::t_neighbors_2d();
      list->d_neighbors = typename AT::t_neighbors_2d();
      list->d_neighbors = typename AT::t_neighbors_2d(Kokkos::NoInit("neighlist:neighbors"), maxatoms, list->maxneighs);
      data.neigh_list.d_neighbors = list->d_neighbors;
      data.neigh_list.maxneighs = list->maxneighs;
      printf("resize d_neighbors to : %d * %d\n", maxatoms, list->maxneighs);

      if (this->force->pair->neigh_sep_strategy == MULTI_NEIGH_LIST) {
        list->maxneighs_outer = data.h_new_maxneighs_outer() * 1.2;
        data.neigh_list.d_neighbors_outer = typename AT::t_neighbors_2d();
        list->d_neighbors_outer = typename AT::t_neighbors_2d();
        list->d_neighbors_outer = typename AT::t_neighbors_2d(Kokkos::NoInit("neighlist:neighbors_outer"), maxatoms, list->maxneighs_outer);
        data.neigh_list.d_neighbors_outer = list->d_neighbors_outer;
        data.neigh_list.maxneighs_outer = list->maxneighs_outer;
        printf("resize d_neighbors_outer to : %d * %d\n", maxatoms, list->maxneighs_outer);
      }
      else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
        list->maxneighs_int2 = data.h_new_maxneighs_int2() * 1.2;
        data.neigh_list.d_neighbors_int2 = typename AT::t_neighbors_2d_int2();
        list->d_neighbors_int2 = typename AT::t_neighbors_2d_int2();
        list->d_neighbors_int2 = typename AT::t_neighbors_2d_int2(Kokkos::NoInit("neighlist:neighbors_int2"), maxatoms, (list->maxneighs_int2 + 1) / 2);
        data.neigh_list.d_neighbors_int2 = list->d_neighbors_int2;
        data.neigh_list.maxneighs_int2 = list->maxneighs_int2;
        printf("resize d_neighbors_int2 to : %d * %d\n", maxatoms, (list->maxneighs_int2 + 1) / 2);
      }

      if (this->force->pair->use_sep_sepcial) {
        list->maxneighs_special = data.h_new_maxneighs_special() * 1.2;
        // printf("resize : resize neighbor special to %d\n", list->maxneighs_special);
        data.neigh_list.d_neighbors_special = typename AT::t_neighbors_2d();
        list->d_neighbors_special = typename AT::t_neighbors_2d();
        list->d_neighbors_special = typename AT::t_neighbors_2d(Kokkos::NoInit("neighlist:neighbors"), maxatoms, list->maxneighs_special);
        data.neigh_list.d_neighbors_special = list->d_neighbors_special;
        data.neigh_list.maxneighs_special = list->maxneighs_special;
        printf("resize d_neighbors_special to : %d * %d\n", maxatoms, list->maxneighs_special);
      }

      if (lmp->kokkos->neigh_transpose) {
        data.neigh_list.d_neighbors_transpose = typename AT::t_neighbors_2d_lr();
        list->d_neighbors_transpose = typename AT::t_neighbors_2d_lr();
        list->d_neighbors_transpose = typename AT::t_neighbors_2d_lr(Kokkos::NoInit("neighlist:neighbors"), maxatoms, list->maxneighs);
        data.neigh_list.d_neighbors_transpose = list->d_neighbors_transpose;
        printf("resize d_neighbors_transpose to : %d * %d\n", maxatoms, list->maxneighs);
      }
    }
  }
  this->force->pair->neighbor_build_time += neighbor_build_timer.seconds();

  if (force->pair->reorder_neighbor == ON_ATOM_SORT) {
    cudaMemcpy(list->numneigh_sort, (list->d_numneigh).data(), nlocal * sizeof(int), cudaMemcpyDeviceToHost);
  }

  if (this->force->pair->neigh_sep_strategy == BASIC_NEIGH_SEP || this->force->pair->neigh_sep_strategy == BASIC_NEIGH_SEP_REV) {
    //printf("neigh transpose : %d\n", lmp->kokkos->neigh_transpose); fflush(stdout);

    int threadsPerBlock = 128;
    int blocksPerGrid = (nall + threadsPerBlock - 1) / threadsPerBlock;

    if (this->force->pair->neigh_sep_strategy == BASIC_NEIGH_SEP_REV) {
      printf("neigh_sort with neigh_sep_strategy BASIC_NEIGH_SEP_REV\n");
      neigh_sort<DeviceType, 1><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
        data.neigh_list.d_numneigh, data.neigh_list.d_neighbors, data.x, force->pair->fhcut_value * force->pair->fhcut_value);
    }
    else {
      // printf("neigh_sort with neigh_sep_strategy BASIC_NEIGH_SEP\n");
      neigh_sort<DeviceType, 0><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
        data.neigh_list.d_numneigh, data.neigh_list.d_neighbors, data.x, force->pair->fhcut_value * force->pair->fhcut_value);
    }
    cudaDeviceSynchronize();

    // printf("finish neigh_sort\n"); fflush(stdout);
    
    // if (this->force->pair->neigh_sep_strategy == BASIC_NEIGH_SEP_REV) {
    //   verify_neigh_sort<DeviceType, 1><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
    //     data.neigh_list.d_numneigh, data.neigh_list.d_neighbors, data.x, force->pair->fhcut_value * force->pair->fhcut_value);
    // }
    // else {
    //   verify_neigh_sort<DeviceType, 0><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
    //     data.neigh_list.d_numneigh, data.neigh_list.d_neighbors, data.x, force->pair->fhcut_value * force->pair->fhcut_value);
    // }
    // cudaDeviceSynchronize();
    // printf("finish verify\n"); fflush(stdout);

    // long long total_neigh_sum = 0;
    // long long fhcut_sum = 0;
    // Kokkos::parallel_reduce(nall, KOKKOS_LAMBDA(const int i, long long &ltotal_neigh_sum, long long &lfhcut_sum) {
    //   ltotal_neigh_sum += data.neigh_list.d_numneigh(i);
    //   lfhcut_sum += data.fhcut_split(i);
    // }, total_neigh_sum, fhcut_sum);
    // Kokkos::fence();
    // printf("total neigh sum / fhcut sum = %lld / %lld\n", total_neigh_sum, fhcut_sum);
  }
  else if (this->force->pair->neigh_sep_strategy == BASIC_NEIGH_SEP_OPT) {
    Kokkos::Timer neighbor_sort_timer;
    neighbor_sort_timer.reset();

    int threadsPerBlock = 128;
    int blocksPerGrid = (nall + threadsPerBlock - 1) / threadsPerBlock;
    // printf("maxneighs : %d\n", data.neigh_list.maxneighs);

    // neigh_sort_opt<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
    //   data.neigh_list.d_numfront, data.neigh_list.d_numback, data.neigh_list.maxneighs, 
    //   data.neigh_list.d_numneigh, data.neigh_list.d_neighbors);
    // cudaDeviceSynchronize();

    neigh_sort_adjust<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
      data.neigh_list.d_numfront, data.neigh_list.d_numback, data.neigh_list.maxneighs, 
      data.neigh_list.d_numneigh, data.neigh_list.d_neighbors);
    cudaDeviceSynchronize();

    // verify_neigh_sort<DeviceType, 0><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
    //   data.neigh_list.d_numneigh, data.neigh_list.d_neighbors, data.x, force->pair->fhcut_value * force->pair->fhcut_value);
    // cudaDeviceSynchronize();

    // static int iter = 0;
    // if (iter == 5) {
    //   FILE* file = fopen("fhcut_diff.txt", "w");
    //   for(int i = 0; i < nall; i++) {
    //     fprintf(file, "%d\n", data.fhcut_split(i) - data.fhcut_split_prev(i));
    //   }
    //   fclose(file);
    // }
    // iter++;

    // auto curr_fhcut_split = data.fhcut_split;
    // auto curr_fhcut_split_prev = data.fhcut_split_prev;
    // Kokkos::parallel_for(nall, KOKKOS_LAMBDA (const int i) {    
    //   curr_fhcut_split_prev(i) = curr_fhcut_split(i);
    // });
    // Kokkos::fence();

    // verify_neigh_sort<DeviceType, 0><<<blocksPerGrid, threadsPerBlock>>>(nall, data.fhcut_split, 
    //   data.neigh_list.d_numneigh, data.neigh_list.d_neighbors, data.x, force->pair->fhcut_value * force->pair->fhcut_value);
    // cudaDeviceSynchronize();
    this->force->pair->neighbor_sort_time += neighbor_sort_timer.seconds();
  }
  else if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH || this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
    Kokkos::fence();
    //printf("begin neigh_exchange\n"); fflush(stdout);
    
    int threadsPerBlock = 128;
    int blocksPerGrid = (nall + threadsPerBlock - 1) / threadsPerBlock;

    // verify_two_end_neigh<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(nall, data.neigh_list.maxneighs,
    //    data.neigh_list.d_numfront, data.neigh_list.d_numback, data.neigh_list.d_neighbors, data.x, 
    //    force->pair->fhcut_value * force->pair->fhcut_value);

    if (this->force->pair->neigh_sep_strategy == TWO_END_NEIGH_INT2) {
      neigh_exchange_int2<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(nall,
        data.neigh_list.d_numneigh, data.neigh_list.d_numneigh_int2, data.neigh_list.d_neighbors, data.neigh_list.d_neighbors_int2, data.x);
      cudaDeviceSynchronize();
    }
    else {
      neigh_exchange<DeviceType><<<blocksPerGrid, threadsPerBlock>>>(nall, data.neigh_list.maxneighs,
        data.neigh_list.d_numfront, data.neigh_list.d_numback, data.neigh_list.d_neighbors, data.x);
      cudaDeviceSynchronize();
    }

    //printf("end neigh_exchange\n"); fflush(stdout);
  }

  if (GHOST) {
    list->inum = atom->nlocal;
    list->gnum = nall - atom->nlocal;
  } else {
    list->inum = nall;
    list->gnum = 0;
  }

  list->k_ilist.template modify<DeviceType>();

  if (lmp->kokkos->neigh_transpose)
    TransposeHelperKokkos<DeviceType, typename AT::t_neighbors_2d,
      typename AT::t_neighbors_2d_lr>(list->d_neighbors, list->d_neighbors_transpose);

  Kokkos::fence();
}

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int NeighborKokkosExecute<DeviceType>::find_special(const int &i, const int &j) const
{
  const int n1 = nspecial(i,0);
  const int n2 = nspecial(i,1);
  const int n3 = nspecial(i,2);

  for (int k = 0; k < n3; k++) {
    if (special(i,k) == tag(j)) {
      if (k < n1) {
        if (special_flag[1] == 0) return -1;
        else if (special_flag[1] == 1) return 0;
        else return 1;
      } else if (k < n2) {
        if (special_flag[2] == 0) return -1;
        else if (special_flag[2] == 1) return 0;
        else return 2;
      } else {
        if (special_flag[3] == 0) return -1;
        else if (special_flag[3] == 1) return 0;
        else return 3;
      }
    }
  }
  return 0;
};

/* ---------------------------------------------------------------------- */

template<class DeviceType>
KOKKOS_INLINE_FUNCTION
int NeighborKokkosExecute<DeviceType>::exclusion(const int &i,const int &j,
                                             const int &itype,const int &jtype) const
{
  int m;

  if (nex_type && ex_type(itype,jtype)) return 1;

  if (nex_group) {
    for (m = 0; m < nex_group; m++) {
      if (mask(i) & ex1_bit(m) && mask(j) & ex2_bit(m)) return 1;
      if (mask(i) & ex2_bit(m) && mask(j) & ex1_bit(m)) return 1;
    }
  }

  if (nex_mol) {
    for (m = 0; m < nex_mol; m++)
      if (ex_mol_intra[m]) { // intra-chain: exclude i-j pair if on same molecule
        if (mask[i] & ex_mol_bit[m] && mask[j] & ex_mol_bit[m] &&
            molecule[i] == molecule[j]) return 1;
      } else                 // exclude i-j pair if on different molecules
        if (mask[i] & ex_mol_bit[m] && mask[j] & ex_mol_bit[m] &&
            molecule[i] != molecule[j]) return 1;
  }

  return 0;
}

/* ---------------------------------------------------------------------- */

template<class DeviceType> template<int HalfNeigh,int Newton,int Tri, int USE_RELATIVE_COORD, int USE_SEP_SPECIAL, int REORDER_NEIGH, NEIGH_SEP_STRATEGY NEIGH_STG>
KOKKOS_FUNCTION
void NeighborKokkosExecute<DeviceType>::
   build_Item(const int &i) const
{
  int n = 0;
  int which = 0;
  int moltemplate;
  if (molecular == Atom::TEMPLATE) moltemplate = 1;
  else moltemplate = 0;
  // get subview of neighbors of i

  const AtomNeighbors neighbors_i = neigh_transpose ?
    neigh_list.get_neighbors_transpose(i) : neigh_list.get_neighbors(i);
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);
  const int itype = type(i);
  tagint itag;
  if (HalfNeigh && Newton && Tri) itag = tag(i);

  const int ibin = c_atom2bin(i);

  const typename ArrayTypes<DeviceType>::t_int_1d_const_um stencil
    = d_stencil;

  // loop over rest of atoms in i's bin, ghosts are at end of linked list
  // if j is owned atom, store it, since j is beyond i in linked list
  // if j is ghost, only store if j coords are "above and to the right" of i

  if (HalfNeigh && Newton && !Tri)
  for (int m = 0; m < c_bincount(ibin); m++) {
    const int j = c_bins(ibin,m);

    if (j <= i) continue;
    if (j >= nlocal) {
      if (x(j,2) < ztmp) continue;
      if (x(j,2) == ztmp) {
        if (x(j,1) < ytmp) continue;
        if (x(j,1) == ytmp && x(j,0) < xtmp) continue;
      }
    }

    const int jtype = type(j);
    if (exclude && exclusion(i,j,itype,jtype)) continue;

    const X_FLOAT delx = xtmp - x(j, 0);
    const X_FLOAT dely = ytmp - x(j, 1);
    const X_FLOAT delz = ztmp - x(j, 2);
    const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;

    if (rsq <= cutneighsq(itype,jtype)) {
      if (molecular != Atom::ATOMIC) {
        if (!moltemplate)
          which = find_special(i,j);
            /* else if (imol >= 0) */
            /*   which = find_special(onemols[imol]->special[iatom], */
            /*                        onemols[imol]->nspecial[iatom], */
            /*                        tag[j]-tagprev); */
            /* else which = 0; */
        if (which == 0) {
          if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
          else n++;
        } else if (minimum_image_check(delx,dely,delz)) {
          if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
          else n++;
        }
        else if (which > 0) {
          if (n < neigh_list.maxneighs) neighbors_i(n++) = j ^ (which << SBBITS);
          else n++;
        }
      } else {
        if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
        else n++;
      }
    }
  }

  for (int k = 0; k < nstencil; k++) {
    const int jbin = ibin + stencil[k];

    int dirx;
    int diry;
    int dirz;
    int dir_neigh_val;
    if(USE_RELATIVE_COORD) {
      dirx = d_stencil_dir(k,0);
      diry = d_stencil_dir(k,1);
      dirz = d_stencil_dir(k,2);
      dir_neigh_val = ((dirx & DIRLOWERMASK) << DIRSHIFTX) | ((diry & DIRLOWERMASK) << DIRSHIFTY) | ((dirz & DIRLOWERMASK) << DIRSHIFTZ);
    }

    if (HalfNeigh && Newton && !Tri && (ibin == jbin)) continue;
    // get subview of jbin
    //const ArrayTypes<DeviceType>::t_int_1d_const_um =Kokkos::subview<t_int_1d_const_um>(bins,jbin,ALL);
      for (int m = 0; m < c_bincount(jbin); m++) {

        const int j = c_bins(jbin,m);
        /// check j&ALLDIRMASK == 0

        if (HalfNeigh && !Newton && j <= i) continue;
        if (!HalfNeigh && j == i) continue;

        // for triclinic, bin stencil is full in all 3 dims
        // must use itag/jtag to eliminate half the I/J interactions
        // cannot use I/J exact coord comparision
        //   b/c transforming orthog -> lambda -> orthog for ghost atoms
        //   with an added PBC offset can shift all 3 coords by epsilon

        if (HalfNeigh && Newton && Tri) {
          if (j <= i) continue;
          if (j >= nlocal) {
            const tagint jtag = tag(j);
            if (itag > jtag) {
              if ((itag+jtag) % 2 == 0) continue;
            } else if (itag < jtag) {
              if ((itag+jtag) % 2 == 1) continue;
            } else {
              if (fabs(x(j,2)-ztmp) > delta) {
                if (x(j,2) < ztmp) continue;
              } else if (fabs(x(j,1)-ytmp) > delta) {
                if (x(j,1) < ytmp) continue;
              } else {
                if (x(j,0) < xtmp) continue;
              }
            }
          }
        }

        const int jtype = type(j);
        if (exclude && exclusion(i,j,itype,jtype)) continue;

        const X_FLOAT delx = xtmp - x(j, 0);
        const X_FLOAT dely = ytmp - x(j, 1);
        const X_FLOAT delz = ztmp - x(j, 2);
        const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq <= cutneighsq(itype,jtype)) {
          int j_hat;
          if(USE_RELATIVE_COORD) {
            j_hat = j ^ dir_neigh_val;
          }
          else {
            j_hat = j;
          }
          if (molecular != Atom::ATOMIC) {
            if (!moltemplate)
              which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
            /* else if (imol >= 0) */
            /*   which = find_special(onemols[imol]->special[iatom], */
            /*                        onemols[imol]->nspecial[iatom], */
            /*                        tag[j]-tagprev); */
            /* else which = 0; */
            if (which == 0) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat;
              else n++;
            } else if (minimum_image_check(delx,dely,delz)) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat;
              else n++;
            }
            else if (which > 0) {
              /// ASSERT failure 
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat ^ (which << SBBITS);
              else n++;
            }
          } else {
            if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat;
            else n++;
          }
        }
      }
  }

  neigh_list.d_numneigh(i) = n;

  if (n > neigh_list.maxneighs) {
    resize() = 1;

    if (n > new_maxneighs()) new_maxneighs() = n; // avoid atomics, safe because in while loop
  }

  neigh_list.d_ilist(i) = i;
}

/* ---------------------------------------------------------------------- */

#ifdef KOKKOS_ENABLE_HIP
#include <hip/hip_version.h>
#if HIP_VERSION_MAJOR < 3 || (HIP_VERSION_MAJOR == 3 && HIP_VERSION_MINOR < 7)
// ROCm versions < 3.7 are missing __syncthreads_count, so we define a functional
// but (probably) not performant workaround
__device__ __forceinline__ int __syncthreads_count(int predicate) {
  __shared__ int test_block[1];
  if (!(threadIdx.x || threadIdx.y || threadIdx.z))
    test_block[0] = 0;
  __syncthreads();
  atomicAdd(test_block, predicate);
  __threadfence_block();
  return test_block[0];
}
#endif
#endif

#ifdef LMP_KOKKOS_GPU
template<class DeviceType> template<int HalfNeigh,int Newton,int Tri, int USE_RELATIVE_COORD, int USE_SEP_SPECIAL, int REORDER_NEIGH, NEIGH_SEP_STRATEGY NEIGH_STG>
LAMMPS_DEVICE_FUNCTION inline
void NeighborKokkosExecute<DeviceType>::build_ItemGPU(typename Kokkos::TeamPolicy<DeviceType>::member_type dev,
                                                      size_t sharedsize) const
{
  auto* sharedmem = static_cast<X_FLOAT *>(dev.team_shmem().get_shmem(sharedsize));

  // loop over atoms in i's bin

  const int atoms_per_bin = c_bins.extent(1);
  const int BINS_PER_TEAM = dev.team_size()/atoms_per_bin <1?1:dev.team_size()/atoms_per_bin;
  const int TEAMS_PER_BIN = atoms_per_bin/dev.team_size()<1?1:atoms_per_bin/dev.team_size();
  const int MY_BIN = dev.team_rank()/atoms_per_bin;

  const int ibin = dev.league_rank()*BINS_PER_TEAM+MY_BIN;

  if (ibin >= mbins) return;

  X_FLOAT* other_x = sharedmem + 6*atoms_per_bin*MY_BIN;
  int* other_id = (int*) &other_x[5 * atoms_per_bin];

  int bincount_current = c_bincount[ibin];

  for (int kk = 0; kk < TEAMS_PER_BIN; kk++) {
    const int MY_II = dev.team_rank()%atoms_per_bin+kk*dev.team_size();
    const int i = MY_II < bincount_current ? c_bins(ibin, MY_II) : -1;

    int n, n_total, n_total_outer, n_total_int2, n_front, n_back, n_total_special;
    int n_inner_front, n_outer_front, prev_fhcut;
    if (NEIGH_STG == TWO_END_NEIGH) {
      n_total = 0;
      n_front = 0;
      n_back = neigh_list.maxneighs - 1;
    }
    else if (NEIGH_STG == BASIC_NEIGH_SEP_OPT) {
      n_inner_front = 0;
      n_back = neigh_list.maxneighs - 1;
    }
    else if (NEIGH_STG == MULTI_NEIGH_LIST) {
      n_total = 0;
      n_total_outer = 0;
    }
    else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
      n_total = 0;
      n_total_int2 = 0;
    }
    else {
      n = 0;
    }
    if (USE_SEP_SPECIAL) {
      n_total_special = 0;
    }

    X_FLOAT xtmp;
    X_FLOAT ytmp;
    X_FLOAT ztmp;
    int itype;
    tagint itag;
    int index = (i >= 0 && i < nlocal) ? i : 0;
    if (REORDER_NEIGH) {
      index = neigh_list.d_neigh_index_inv[index];
    }
    if (NEIGH_STG == BASIC_NEIGH_SEP_OPT) {
      prev_fhcut = fhcut_split(index);
      n_total = prev_fhcut;
      n_outer_front = prev_fhcut;
    }
    const AtomNeighbors neighbors_i = neigh_transpose ?
    neigh_list.get_neighbors_transpose(index) : neigh_list.get_neighbors(index);

    AtomNeighbors neighbors_i_outer;
    AtomNeighbors_int2 neighbors_i_int2;
    if (NEIGH_STG == MULTI_NEIGH_LIST) {
      neighbors_i_outer = neigh_list.get_neighbors_outer(index);
    } 
    else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
      neighbors_i_int2 = neigh_list.get_neighbors_int2(index);
    } 
    AtomNeighbors neighbors_i_special;
    if (USE_SEP_SPECIAL) {
      neighbors_i_special = neigh_list.get_neighbors_special(index);
    }

    if (i >= 0) {
      xtmp = x(i, 0);
      ytmp = x(i, 1);
      ztmp = x(i, 2);
      itype = type(i);
      other_x[MY_II] = xtmp;
      other_x[MY_II + atoms_per_bin] = ytmp;
      other_x[MY_II + 2 * atoms_per_bin] = ztmp;
      other_x[MY_II + 3 * atoms_per_bin] = itype;
      if (HalfNeigh && Newton && Tri) {
        itag = tag(i);
        other_x[MY_II + 4 * atoms_per_bin] = itag;
      }
    }
    other_id[MY_II] = i;

#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int test = (__syncthreads_count(i >= 0 && i < nlocal) == 0);
    if (test) return;
#elif defined(KOKKOS_ENABLE_SYCL)
    int not_done = (i >= 0 && i < nlocal);
    dev.team_reduce(Kokkos::Max<int>(not_done));
    if(not_done == 0) return;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
    dev.team_barrier();
#endif

  // loop over rest of atoms in i's bin, ghosts are at end of linked list
  // if j is owned atom, store it, since j is beyond i in linked list
  // if j is ghost, only store if j coords are "above and to the right" of i

    if (HalfNeigh && Newton && !Tri)
    if (i >= 0 && i < nlocal) {
      #pragma unroll 4
      for (int m = 0; m < bincount_current; m++) {
        int j = other_id[m];

        if (j <= i) continue;
        if (j >= nlocal) {
          if (x(j,2) < ztmp) continue;
          if (x(j,2) == ztmp) {
            if (x(j,1) < ytmp) continue;
            if (x(j,1) == ytmp && x(j,0) < xtmp) continue;
          }
        }

        const int jtype = other_x[m + 3 * atoms_per_bin];
        if (exclude && exclusion(i,j,itype,jtype)) continue;

        const X_FLOAT delx = xtmp - other_x[m];
        const X_FLOAT dely = ytmp - other_x[m + atoms_per_bin];
        const X_FLOAT delz = ztmp - other_x[m + 2 * atoms_per_bin];
        const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq <= cutneighsq(itype,jtype)) {

          if(USE_RELATIVE_COORD) {
            // j is in the same bin as i, no need to assign direction
          }

          if (NEIGH_STG == MULTI_NEIGH_LIST) {
            if (molecular != Atom::ATOMIC) {
              int which = 0;
              if (!moltemplate)
                which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
              /* else if (imol >= 0) */
              /*   which = find_special(onemols[imol]->special[iatom], */
              /*                        onemols[imol]->nspecial[iatom], */
              /*                        tag[j]-tagprev); */
              /* else which = 0; */
              if (which == 0) {
                if(rsq > fhcut_rsq) {
                  if(n_total_outer < neigh_list.maxneighs_outer)
                    neighbors_i_outer(n_total_outer++) = j;
                  else n_total_outer++;
                }
                else {
                  if (n_total < neigh_list.maxneighs) 
                    neighbors_i(n_total++) = j;
                  else n_total++;
                } 
              } else if (minimum_image_check(delx,dely,delz)) {
                if(rsq > fhcut_rsq) {
                  if(n_total_outer < neigh_list.maxneighs_outer)
                    neighbors_i_outer(n_total_outer++) = j;
                  else n_total_outer++;
                }
                else {
                  if (n_total < neigh_list.maxneighs) 
                    neighbors_i(n_total++) = j;
                  else n_total++;
                } 
              }
              else if (which > 0) {
                if (USE_SEP_SPECIAL) {
                  if (n_total_special < neigh_list.maxneighs_special) {
                    neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                  }
                  else n_total_special++;
                }
                else {
                  if(rsq > fhcut_rsq) {
                    if(n_total_outer < neigh_list.maxneighs_outer)
                      neighbors_i_outer(n_total_outer++) = j;
                    else n_total_outer++;
                  }
                  else {
                    if (n_total < neigh_list.maxneighs) 
                      neighbors_i(n_total++) = j ^ (which << SBBITS);
                    else n_total++;
                  } 
                }
              }
            } else {
              if(rsq > fhcut_rsq) {
                if(n_total_outer < neigh_list.maxneighs_outer)
                    neighbors_i_outer(n_total_outer++) = j;
                else n_total_outer++;
              }
              else {
                if (n_total < neigh_list.maxneighs) 
                  neighbors_i(n_total++) = j;
                else n_total++;
              }
            }
          }
          else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
            if (molecular != Atom::ATOMIC) {
              int which = 0;
              if (!moltemplate)
                which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
              /* else if (imol >= 0) */
              /*   which = find_special(onemols[imol]->special[iatom], */
              /*                        onemols[imol]->nspecial[iatom], */
              /*                        tag[j]-tagprev); */
              /* else which = 0; */
              if (which == 0) {
                if(rsq > fhcut_rsq) {
                  if(n_total_int2 < neigh_list.maxneighs_int2) {
                    if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j;
                    else neighbors_i_int2(n_total_int2 >> 1).x = j;
                  }
                  n_total_int2++;
                }
                else {
                  if (n_total < neigh_list.maxneighs) 
                    neighbors_i(n_total++) = j;
                  else n_total++;
                } 
              } else if (minimum_image_check(delx,dely,delz)) {
                if(rsq > fhcut_rsq) {
                  if(n_total_int2 < neigh_list.maxneighs_int2) {
                    if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j;
                    else neighbors_i_int2(n_total_int2 >> 1).x = j;
                  }
                  n_total_int2++;
                }
                else {
                  if (n_total < neigh_list.maxneighs) 
                    neighbors_i(n_total++) = j;
                  else n_total++;
                } 
              }
              else if (which > 0) {
                if (USE_SEP_SPECIAL) {
                  if (n_total_special < neigh_list.maxneighs_special) {
                    neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                  }
                  else n_total_special++;
                }
                else {
                  if(rsq > fhcut_rsq) {
                    if(n_total_int2 < neigh_list.maxneighs_int2) {
                      if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j ^ (which << SBBITS);
                      else neighbors_i_int2(n_total_int2 >> 1).x = j ^ (which << SBBITS);
                    }
                    n_total_int2++;
                  }
                  else {
                    if (n_total < neigh_list.maxneighs) 
                      neighbors_i(n_total++) = j ^ (which << SBBITS);
                    else n_total++;
                  } 
                }
              }
            } else {
              if(rsq > fhcut_rsq) {
                if(n_total_int2 < neigh_list.maxneighs_int2) {
                  if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j;
                  else neighbors_i_int2(n_total_int2 >> 1).x = j;
                }
                n_total_int2++;
              }
              else {
                if (n_total < neigh_list.maxneighs) 
                  neighbors_i(n_total++) = j;
                else n_total++;
              }
            }
          }
          else if(NEIGH_STG == TWO_END_NEIGH) {
            if (molecular != Atom::ATOMIC) {
              int which = 0;
              if (!moltemplate)
                which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
              /* else if (imol >= 0) */
              /*   which = find_special(onemols[imol]->special[iatom], */
              /*                        onemols[imol]->nspecial[iatom], */
              /*                        tag[j]-tagprev); */
              /* else which = 0; */
              if (which == 0) {
                if (n_total < neigh_list.maxneighs) {
                  if(rsq > fhcut_rsq)
                    neighbors_i(n_front++) = j;
                  else neighbors_i(n_back--) = j;
                }
                n_total++;
              } else if (minimum_image_check(delx,dely,delz)) {
                if (n_total < neigh_list.maxneighs) {
                  if(rsq > fhcut_rsq)
                    neighbors_i(n_front++) = j;
                  else neighbors_i(n_back--) = j;
                }
                n_total++;
              }
              else if (which > 0) {
                if (USE_SEP_SPECIAL) {
                  if (n_total_special < neigh_list.maxneighs_special) {
                    neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                  }
                  else n_total_special++;
                }
                else {
                  if (n_total < neigh_list.maxneighs) {
                    if(rsq > fhcut_rsq)
                      neighbors_i(n_front++) = j ^ (which << SBBITS);
                    else neighbors_i(n_back--) = j ^ (which << SBBITS);
                  }
                  n_total++;
                }
              }
            } else {
              if (n_total < neigh_list.maxneighs) {
                if(rsq > fhcut_rsq)
                  neighbors_i(n_front++) = j;
                else neighbors_i(n_back--) = j;
              }
              n_total++;
            }
          }
          else if(NEIGH_STG == BASIC_NEIGH_SEP_OPT) {
            int neigh_flag = 0;
            int neigh_val;
            if (molecular != Atom::ATOMIC) {
              int which = 0;
              if (!moltemplate)
                which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
              /* else if (imol >= 0) */
              /*   which = find_special(onemols[imol]->special[iatom], */
              /*                        onemols[imol]->nspecial[iatom], */
              /*                        tag[j]-tagprev); */
              /* else which = 0; */
              if (which == 0) {
                neigh_flag = 1; neigh_val = j;
                // if (n_total < neigh_list.maxneighs) {
                //   if(rsq <= fhcut_rsq)
                //     neighbors_i(n_front++) = j;
                //   else neighbors_i(n_back--) = j;
                // }
                // n_total++;
              } else if (minimum_image_check(delx,dely,delz)) {
                neigh_flag = 1; neigh_val = j;
                // if (n_total < neigh_list.maxneighs) {
                //   if(rsq <= fhcut_rsq)
                //     neighbors_i(n_front++) = j;
                //   else neighbors_i(n_back--) = j;
                // }
                // n_total++;
              }
              else if (which > 0) {
                if (USE_SEP_SPECIAL) {
                  if (n_total_special < neigh_list.maxneighs_special) {
                    neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                  }
                  else n_total_special++;
                }
                else {
                  neigh_flag = 1; neigh_val = j ^ (which << SBBITS);
                  // if (n_total < neigh_list.maxneighs) {
                  //   if(rsq <= fhcut_rsq)
                  //     neighbors_i(n_front++) = j ^ (which << SBBITS);
                  //   else neighbors_i(n_back--) = j ^ (which << SBBITS);
                  // }
                  // n_total++;
                }
              }
            } else {
              neigh_flag = 1; neigh_val = j;
              // if (n_total < neigh_list.maxneighs) {
              //   if(rsq <= fhcut_rsq)
              //     neighbors_i(n_front++) = j;
              //   else neighbors_i(n_back--) = j;
              // }
              // n_total++;
            }
            if (neigh_flag) {
              if (n_total < neigh_list.maxneighs) {
                int neigh_pos;
                if (rsq <= fhcut_rsq) {
                  if (n_inner_front < prev_fhcut) {
                    neigh_pos = n_inner_front++;
                    // neighbors_i(n_inner_front++) = neigh_val;
                  }
                  else {
                    // int t = neighbors_i(n_front);
                    // neighbors_i(n_total++) = t;
                    // neighbors_i(n_front++) = neigh_val;
                    neigh_pos = n_back--;
                    // neighbors_i(n_back--) = neigh_val;
                    n_total++;
                  }
                }
                else {
                  neigh_pos = n_outer_front++;
                  // neighbors_i(n_outer_front++) = neigh_val;
                  n_total++;
                }
                neighbors_i(neigh_pos) = neigh_val;
              }
              else {
                n_total++;
              } 
            }
          }
          else {
            if (molecular != Atom::ATOMIC) {
              int which = 0;
              if (!moltemplate)
                which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
              /* else if (imol >= 0) */
              /*   which = find_special(onemols[imol]->special[iatom], */
              /*                        onemols[imol]->nspecial[iatom], */
              /*                        tag[j]-tagprev); */
              /* else which = 0; */
              if (which == 0) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
                else n++;
              } else if (minimum_image_check(delx,dely,delz)) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
                else n++;
              }
              else if (which > 0) {
                if (USE_SEP_SPECIAL) {
                  if (n_total_special < neigh_list.maxneighs_special) {
                    neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                  }
                  else n_total_special++;
                }
                else {
                  if (n < neigh_list.maxneighs) neighbors_i(n++) = j ^ (which << SBBITS);
                  else n++;
                }
              }
            } else {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
              else n++;
            }
          }
        }

      }
    }
    dev.team_barrier();

    const typename ArrayTypes<DeviceType>::t_int_1d_const_um stencil
      = d_stencil;
    for (int k = 0; k < nstencil; k++) {
      const int jbin = ibin + stencil[k];

      int dirx;
      int diry;
      int dirz;
      int dir_neigh_val;
      if (USE_RELATIVE_COORD) {
        dirx = d_stencil_dir(k,0);
        diry = d_stencil_dir(k,1);
        dirz = d_stencil_dir(k,2);
        dir_neigh_val = ((dirx & DIRLOWERMASK) << DIRSHIFTX) | ((diry & DIRLOWERMASK) << DIRSHIFTY) | ((dirz & DIRLOWERMASK) << DIRSHIFTZ);
      }

      if (HalfNeigh && Newton && !Tri && (ibin == jbin)) continue;

      bincount_current = c_bincount[jbin];
      int j = MY_II < bincount_current ? c_bins(jbin, MY_II) : -1;

      if (j >= 0) {
        other_x[MY_II] = x(j, 0);
        other_x[MY_II + atoms_per_bin] = x(j, 1);
        other_x[MY_II + 2 * atoms_per_bin] = x(j, 2);
        other_x[MY_II + 3 * atoms_per_bin] = type(j);
        if (HalfNeigh && Newton && Tri)
          other_x[MY_II + 4 * atoms_per_bin] = tag(j);
      }

      other_id[MY_II] = j;

      dev.team_barrier();

      if (i >= 0 && i < nlocal) {
        #pragma unroll 8
        for (int m = 0; m < bincount_current; m++) {
          const int j = other_id[m];

          if (HalfNeigh && !Newton && j <= i) continue;
          if (!HalfNeigh && j == i) continue;

          // for triclinic, bin stencil is full in all 3 dims
          // must use itag/jtag to eliminate half the I/J interactions
          // cannot use I/J exact coord comparision
          //   b/c transforming orthog -> lambda -> orthog for ghost atoms
          //   with an added PBC offset can shift all 3 coords by epsilon

          if (HalfNeigh && Newton && Tri) {
            if (j <= i) continue;
            if (j >= nlocal) {
              const tagint jtag = other_x[m + 4 * atoms_per_bin];
              if (itag > jtag) {
                if ((itag+jtag) % 2 == 0) continue;
              } else if (itag < jtag) {
                if ((itag+jtag) % 2 == 1) continue;
              } else {
                if (fabs(x(j,2)-ztmp) > delta) {
                  if (x(j,2) < ztmp) continue;
                } else if (fabs(x(j,1)-ytmp) > delta) {
                  if (x(j,1) < ytmp) continue;
                } else {
                  if (x(j,0) < xtmp) continue;
                }
              }
            }
          }

          const int jtype = other_x[m + 3 * atoms_per_bin];
          if (exclude && exclusion(i,j,itype,jtype)) continue;

          const X_FLOAT delx = xtmp - other_x[m];
          const X_FLOAT dely = ytmp - other_x[m + atoms_per_bin];
          const X_FLOAT delz = ztmp - other_x[m + 2 * atoms_per_bin];
          const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;

          if (rsq <= cutneighsq(itype,jtype)) {
            // if (i == 0) {
            //   printf("build_Item i %d, j %d, dirx %d, diry %d, dirz %d ", i, j, dirx, diry, dirz);
            //   printf("x %f, y %f, z %f, xn %f, yn %f, zn %f\n", xtmp, ytmp, ztmp, other_x[m], other_x[m + atoms_per_bin], other_x[m + 2 * atoms_per_bin]);
            
            //   const int ibin = c_atom2bin(i);
            //   int ix = ibin % mbinx;
            //   int iy = (ibin/mbinx) % mbiny;
            //   int iz = ibin/(mbinx*mbiny);

            //   const int jbin = c_atom2bin(j);
            //   int jx = jbin % mbinx;
            //   int jy = (jbin/mbinx) % mbiny;
            //   int jz = jbin/(mbinx*mbiny);

            //   printf("ibin : %d, %d, %d, jbin : %d, %d, %d\n", ix, iy, iz, jx, jy, jz);
            // } 
            int j_hat;
            if (USE_RELATIVE_COORD) {
              // // debug relative coord
              // if (i == 0) {
              //   const float a_delx = x(i, 0) - x(j, 0);
              //   const float a_dely = x(i, 1) - x(j, 1);
              //   const float a_delz = x(i, 2) - x(j, 2);    

              //   const float t_delx = x_rel(i, 0) - x_rel(j, 0) - dirx * binsizex;
              //   const float t_dely = x_rel(i, 1) - x_rel(j, 1) - diry * binsizey;
              //   const float t_delz = x_rel(i, 2) - x_rel(j, 2) - dirz * binsizez;
                
              //   if (fabs(a_delx - t_delx) > 0.0001 || fabs(a_dely - t_dely) > 0.0001 || fabs(a_delz - t_delz) > 0.0001) {
              //     printf("build Error coord: %d %d %f %f %f %f %f %f\n", i, j, a_delx, a_dely, a_delz, t_delx, t_dely, t_delz);
              //     printf("build bin size : %f %f %f\n", binsizex, binsizey, binsizez);
              //     printf("build dir mask : %d %d %d\n", dirx, diry, dirz);
              //     printf("build relative coord : (%f %f %f), (%f %f %f)\n", x_rel(i, 0), x_rel(i, 1), x_rel(i, 2), x_rel(j, 0), x_rel(j, 1), x_rel(j, 2));
              //     printf("build abs coord : (%f %f %f), (%f %f %f)\n", x(i, 0), x(i, 1), x(i, 2), x(j, 0), x(j, 1), x(j, 2));
              //   }
              // }
              j_hat = j ^ dir_neigh_val;
            }
            else {
              j_hat = j;
            }

            if (NEIGH_STG == MULTI_NEIGH_LIST) {
              if (molecular != Atom::ATOMIC) {
                int which = 0;
                if (!moltemplate)
                  which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
                /* else if (imol >= 0) */
                /*   which = find_special(onemols[imol]->special[iatom], */
                /*                        onemols[imol]->nspecial[iatom], */
                /*                        tag[j]-tagprev); */
                /* else which = 0; */
                if (which == 0) {
                  if(rsq > fhcut_rsq) {
                    if(n_total_outer < neigh_list.maxneighs_outer)
                      neighbors_i_outer(n_total_outer++) = j_hat;
                    else n_total_outer++;
                  }
                  else {
                    if (n_total < neigh_list.maxneighs) 
                      neighbors_i(n_total++) = j_hat;
                    else n_total++;
                  } 
                } else if (minimum_image_check(delx,dely,delz)) {
                  if(rsq > fhcut_rsq) {
                    if(n_total_outer < neigh_list.maxneighs_outer)
                      neighbors_i_outer(n_total_outer++) = j_hat;
                    else n_total_outer++;
                  }
                  else {
                    if (n_total < neigh_list.maxneighs) 
                      neighbors_i(n_total++) = j_hat;
                    else n_total++;
                  } 
                }
                else if (which > 0) {
                  if (USE_SEP_SPECIAL) {
                    if (n_total_special < neigh_list.maxneighs_special) {
                      neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                    }
                    else n_total_special++;
                  }
                  else {
                    /// ASSERT failure
                    if(rsq > fhcut_rsq) {
                      if(n_total_outer < neigh_list.maxneighs_outer)
                        neighbors_i_outer(n_total_outer++) = j_hat ^ (which << SBBITS);
                      else n_total_outer++;
                    }
                    else {
                      if (n_total < neigh_list.maxneighs) 
                        neighbors_i(n_total++) = j_hat ^ (which << SBBITS);
                      else n_total++;
                    } 
                  }
                }
              } else {
                if(rsq > fhcut_rsq) {
                  if(n_total_outer < neigh_list.maxneighs_outer)
                    neighbors_i_outer(n_total_outer++) = j_hat;
                  else n_total_outer++;
                }
                else {
                  if (n_total < neigh_list.maxneighs) 
                    neighbors_i(n_total++) = j_hat;
                  else n_total++;
                } 
              }
            }
            else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
              if (molecular != Atom::ATOMIC) {
                int which = 0;
                if (!moltemplate)
                  which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
                /* else if (imol >= 0) */
                /*   which = find_special(onemols[imol]->special[iatom], */
                /*                        onemols[imol]->nspecial[iatom], */
                /*                        tag[j]-tagprev); */
                /* else which = 0; */
                if (which == 0) {
                  if(rsq > fhcut_rsq) {
                    if(n_total_int2 < neigh_list.maxneighs_int2) {
                      if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j_hat;
                      else neighbors_i_int2(n_total_int2 >> 1).x = j_hat;
                    }
                    n_total_int2++;
                  }
                  else {
                    if (n_total < neigh_list.maxneighs) 
                      neighbors_i(n_total++) = j_hat;
                    else n_total++;
                  } 
                } else if (minimum_image_check(delx,dely,delz)) {
                  if(rsq > fhcut_rsq) {
                    if(n_total_int2 < neigh_list.maxneighs_int2) {
                      if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j_hat;
                      else neighbors_i_int2(n_total_int2 >> 1).x = j_hat;
                    }
                    n_total_int2++;
                  }
                  else {
                    if (n_total < neigh_list.maxneighs) 
                      neighbors_i(n_total++) = j_hat;
                    else n_total++;
                  } 
                }
                else if (which > 0) {
                  if (USE_SEP_SPECIAL) {
                    if (n_total_special < neigh_list.maxneighs_special) {
                      neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                    }
                    else n_total_special++;
                  }
                  else {
                    /// ASSERT failure
                    if(rsq > fhcut_rsq) {
                      if(n_total_int2 < neigh_list.maxneighs_int2) {
                        if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j_hat ^ (which << SBBITS);
                        else neighbors_i_int2(n_total_int2 >> 1).x = j_hat ^ (which << SBBITS);
                      }
                      n_total_int2++;
                    }
                    else {
                      if (n_total < neigh_list.maxneighs) 
                        neighbors_i(n_total++) = j_hat ^ (which << SBBITS);
                      else n_total++;
                    } 
                  }
                }
              } else {
                if(rsq > fhcut_rsq) {
                  if(n_total_int2 < neigh_list.maxneighs_int2) {
                    if(n_total_int2 & 1) neighbors_i_int2(n_total_int2 >> 1).y = j_hat;
                    else neighbors_i_int2(n_total_int2 >> 1).x = j_hat;
                  }
                  n_total_int2++;
                }
                else {
                  if (n_total < neigh_list.maxneighs) 
                    neighbors_i(n_total++) = j_hat;
                  else n_total++;
                } 
              }
            }
            else if(NEIGH_STG == TWO_END_NEIGH) {
              if (molecular != Atom::ATOMIC) {
                int which = 0;
                if (!moltemplate)
                  which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
                /* else if (imol >= 0) */
                /*   which = find_special(onemols[imol]->special[iatom], */
                /*                        onemols[imol]->nspecial[iatom], */
                /*                        tag[j]-tagprev); */
                /* else which = 0; */
                if (which == 0) {
                  if (n_total < neigh_list.maxneighs) {
                    if(rsq > fhcut_rsq)
                      neighbors_i(n_front++) = j_hat;
                    else neighbors_i(n_back--) = j_hat;
                  }
                  n_total++;
                } else if (minimum_image_check(delx,dely,delz)) {
                  if (n_total < neigh_list.maxneighs) {
                    if(rsq > fhcut_rsq)
                      neighbors_i(n_front++) = j_hat;
                    else neighbors_i(n_back--) = j_hat;
                  }
                  n_total++;
                }
                else if (which > 0) {
                  if (USE_SEP_SPECIAL) {
                    if (n_total_special < neigh_list.maxneighs_special) {
                      neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                    }
                    else n_total_special++;
                  }
                  else {
                    /// ASSERT failure
                    if (n_total < neigh_list.maxneighs) {
                      if(rsq > fhcut_rsq)
                        neighbors_i(n_front++) = j_hat ^ (which << SBBITS);
                      else neighbors_i(n_back--) = j_hat ^ (which << SBBITS);
                    }
                    n_total++;
                  }
                }
              } else {
                if (n_total < neigh_list.maxneighs) {
                  if(rsq > fhcut_rsq)
                    neighbors_i(n_front++) = j_hat;
                  else neighbors_i(n_back--) = j_hat;
                }
                n_total++;
              }
            }
            else if(NEIGH_STG == BASIC_NEIGH_SEP_OPT) {
              int neigh_flag = 0;
              int neigh_val;
              if (molecular != Atom::ATOMIC) {
                int which = 0;
                if (!moltemplate)
                  which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
                /* else if (imol >= 0) */
                /*   which = find_special(onemols[imol]->special[iatom], */
                /*                        onemols[imol]->nspecial[iatom], */
                /*                        tag[j]-tagprev); */
                /* else which = 0; */
                if (which == 0) {
                  neigh_flag = 1; neigh_val = j_hat;
                  // if (n_total < neigh_list.maxneighs) {
                  //   if(rsq <= fhcut_rsq)
                  //     neighbors_i(n_front++) = j_hat;
                  //   else neighbors_i(n_back--) = j_hat;
                  // }
                  // n_total++;
                } else if (minimum_image_check(delx,dely,delz)) {
                  neigh_flag = 1; neigh_val = j_hat;
                  // if (n_total < neigh_list.maxneighs) {
                  //   if(rsq <= fhcut_rsq)
                  //     neighbors_i(n_front++) = j_hat;
                  //   else neighbors_i(n_back--) = j_hat;
                  // }
                  // n_total++;
                }
                else if (which > 0) {
                  if (USE_SEP_SPECIAL) {
                    if (n_total_special < neigh_list.maxneighs_special) {
                      neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                    }
                    else n_total_special++;
                  }
                  else {
                    neigh_flag = 1; neigh_val = j_hat ^ (which << SBBITS);
                    /// ASSERT failure
                    // if (n_total < neigh_list.maxneighs) {
                    //   if(rsq <= fhcut_rsq)
                    //     neighbors_i(n_front++) = j_hat ^ (which << SBBITS);
                    //   else neighbors_i(n_back--) = j_hat ^ (which << SBBITS);
                    // }
                    // n_total++;
                  }
                }
              } else {
                neigh_flag = 1; neigh_val = j_hat;
                // if (n_total < neigh_list.maxneighs) {
                //   if(rsq <= fhcut_rsq)
                //     neighbors_i(n_front++) = j_hat;
                //   else neighbors_i(n_back--) = j_hat;
                // }
                // n_total++;
              }
              if (neigh_flag) {
                if (n_total < neigh_list.maxneighs) {
                  int neigh_pos;
                  if (rsq <= fhcut_rsq) {
                    if (n_inner_front < prev_fhcut) {
                      neigh_pos = n_inner_front++;
                      // neighbors_i(n_inner_front++) = neigh_val;
                    }
                    else {
                      // int t = neighbors_i(n_front);
                      // neighbors_i(n_total++) = t;
                      // neighbors_i(n_front++) = neigh_val;
                      neigh_pos = n_back--;
                      // neighbors_i(n_back--) = neigh_val;
                      n_total++;
                    }
                  }
                  else {
                    neigh_pos = n_outer_front++;
                    // neighbors_i(n_outer_front++) = neigh_val;
                    n_total++;
                  }
                  neighbors_i(neigh_pos) = neigh_val;
                }
                else {
                  n_total++;
                } 
              }
            }
            else {
              if (molecular != Atom::ATOMIC) {
                int which = 0;
                if (!moltemplate)
                  which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
                /* else if (imol >= 0) */
                /*   which = find_special(onemols[imol]->special[iatom], */
                /*                        onemols[imol]->nspecial[iatom], */
                /*                        tag[j]-tagprev); */
                /* else which = 0; */
                if (which == 0) {
                  if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat;
                  else n++;
                } else if (minimum_image_check(delx,dely,delz)) {
                  if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat;
                  else n++;
                }
                else if (which > 0) {
                  if (USE_SEP_SPECIAL) {
                    if (n_total_special < neigh_list.maxneighs_special) {
                      neighbors_i_special(n_total_special++) = j ^ (which << SBBITS);
                    }
                    else n_total_special++;
                  }
                  else {
                    /// ASSERT failure
                    if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat ^ (which << SBBITS);
                    else n++;
                  }
                }
              } else {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = j_hat;
                else n++;
              }
            }
          }

        }
      }
      dev.team_barrier();
    }

    if (NEIGH_STG == MULTI_NEIGH_LIST) {
      if (i >= 0 && i < nlocal) {
        neigh_list.d_numneigh_outer(i) = n_total_outer;
        neigh_list.d_numneigh(i) = n_total;
        neigh_list.d_ilist(i) = i;
        if (USE_SEP_SPECIAL) {
          neigh_list.d_numneigh_special(i) = n_total_special;
        }
      }

      if (n_total + 1 > neigh_list.maxneighs) {
        resize() = 1;
        if (n_total + 1 > new_maxneighs()) new_maxneighs() = n_total + 1; // avoid atomics, safe because in while loop
      }
      if (n_total_outer > neigh_list.maxneighs_outer) {
        resize() = 1;
        if (n_total_outer > new_maxneighs_outer()) new_maxneighs_outer() = n_total_outer; // avoid atomics, safe because in while loop
      }
      if (USE_SEP_SPECIAL && n_total_special > neigh_list.maxneighs_special) {
        resize() = 1;
        if (n_total_special > new_maxneighs_special()) new_maxneighs_special() = n_total_special;
      }
    }
    else if (NEIGH_STG == TWO_END_NEIGH_INT2) {
      if (i >= 0 && i < nlocal) {
        neigh_list.d_numneigh_int2(i) = n_total_int2;
        neigh_list.d_numneigh(i) = n_total;
        neigh_list.d_ilist(i) = i;
        if (USE_SEP_SPECIAL) {
          neigh_list.d_numneigh_special(i) = n_total_special;
        }
      }

      if (n_total + 1 > neigh_list.maxneighs) {
        resize() = 1;
        if (n_total + 1 > new_maxneighs()) new_maxneighs() = n_total + 1; // avoid atomics, safe because in while loop
      }
      if (n_total_int2 > neigh_list.maxneighs_int2) {
        resize() = 1;
        if (n_total_int2 > new_maxneighs_int2()) new_maxneighs_int2() = n_total_int2; // avoid atomics, safe because in while loop
      }
      if (USE_SEP_SPECIAL && n_total_special > neigh_list.maxneighs_special) {
        resize() = 1;
        if (n_total_special > new_maxneighs_special()) new_maxneighs_special() = n_total_special;
      }
    }
    else if(NEIGH_STG == TWO_END_NEIGH) {
      if (i >= 0 && i < nlocal) {
        neigh_list.d_numfront(i) = n_front;
        neigh_list.d_numback(i) = n_back + 1;
        neigh_list.d_ilist(i) = i;
        if (USE_SEP_SPECIAL) {
          neigh_list.d_numneigh_special(i) = n_total_special;
        }
      }

      if (n_total > neigh_list.maxneighs) {
        resize() = 1;
        if (n_total > new_maxneighs()) new_maxneighs() = n_total; // avoid atomics, safe because in while loop
      }
      if (USE_SEP_SPECIAL && n_total_special > neigh_list.maxneighs_special) {
        resize() = 1;
        if (n_total_special > new_maxneighs_special()) new_maxneighs_special() = n_total_special;
      }
    }
    else if(NEIGH_STG == BASIC_NEIGH_SEP_OPT) {
      if (i >= 0 && i < nlocal) {
        neigh_list.d_numfront(i) = n_inner_front;
        neigh_list.d_numback(i) = n_back + 1;
        neigh_list.d_numneigh(i) = n_outer_front;
        neigh_list.d_ilist(i) = i;
        if (USE_SEP_SPECIAL) {
          neigh_list.d_numneigh_special(i) = n_total_special;
        }
      }

      if (n_total > neigh_list.maxneighs) {
        resize() = 1;
        if (n_total > new_maxneighs()) new_maxneighs() = n_total; // avoid atomics, safe because in while loop
      }
      if (USE_SEP_SPECIAL && n_total_special > neigh_list.maxneighs_special) {
        resize() = 1;
        if (n_total_special > new_maxneighs_special()) new_maxneighs_special() = n_total_special;
      }
    }
    else {
      if (i >= 0 && i < nlocal) {
        neigh_list.d_numneigh(i) = n;
        neigh_list.d_ilist(i) = i;
        if (USE_SEP_SPECIAL) {
          neigh_list.d_numneigh_special(i) = n_total_special;
        }
      }

      if (n > neigh_list.maxneighs) {
        resize() = 1;
        if (n > new_maxneighs()) new_maxneighs() = n; // avoid atomics, safe because in while loop
      }
      if (USE_SEP_SPECIAL && n_total_special > neigh_list.maxneighs_special) {
        resize() = 1;
        if (n_total_special > new_maxneighs_special()) new_maxneighs_special() = n_total_special;
      }
    }
  }
}
#endif

// template<class DeviceType>
// void NeighborKokkosExecute<DeviceType>::build_ReleativeX(const int &i) const {
//   const X_FLOAT xtmp = x(i, 0);
//   const X_FLOAT ytmp = x(i, 1);
//   const X_FLOAT ztmp = x(i, 2);

//   const int ibin = c_atom2bin(i);

//   X_FLOAT basex, basey, basez;
//   bin_base_coord(basex, basey, basez, ibin);

//   x_rel(i, 0) = xtmp - basex;
//   x_rel(i, 1) = ytmp - basey;
//   x_rel(i, 2) = ztmp - basez;
// }

/* ---------------------------------------------------------------------- */

template<class DeviceType>  template<int HalfNeigh>
KOKKOS_FUNCTION
void NeighborKokkosExecute<DeviceType>::
   build_ItemGhost(const int &i) const
{
  /* if necessary, goto next page and add pages */
  int n = 0;
  int which = 0;
  int moltemplate;
  if (molecular == Atom::TEMPLATE) moltemplate = 1;
  else moltemplate = 0;
  // get subview of neighbors of i

  const AtomNeighbors neighbors_i = neigh_transpose ?
    neigh_list.get_neighbors_transpose(i) : neigh_list.get_neighbors(i);
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);
  const int itype = type(i);

  const typename ArrayTypes<DeviceType>::t_int_1d_const_um stencil
    = d_stencil;
  const typename ArrayTypes<DeviceType>::t_int_1d_3_const_um stencilxyz
    = d_stencilxyz;

  // loop over all atoms in surrounding bins in stencil including self
  // when i is a ghost atom, must check if stencil bin is out of bounds
  // skip i = j
  // no molecular test when i = ghost atom

  if (i < nlocal) {
    const int ibin = c_atom2bin(i);
    for (int k = 0; k < nstencil; k++) {
      const int jbin = ibin + stencil[k];
      for (int m = 0; m < c_bincount(jbin); m++) {
        const int j = c_bins(jbin,m);

        if (HalfNeigh && j <= i) continue;
        else if (j == i) continue;

        const int jtype = type[j];
        if (exclude && exclusion(i,j,itype,jtype)) continue;

        const X_FLOAT delx = xtmp - x(j,0);
        const X_FLOAT dely = ytmp - x(j,1);
        const X_FLOAT delz = ztmp - x(j,2);
        const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq <= cutneighsq(itype,jtype)) {
          if (molecular != Atom::ATOMIC) {
            if (!moltemplate)
              which = find_special(i,j);
            /* else if (imol >= 0) */
            /*   which = find_special(onemols[imol]->special[iatom], */
            /*                        onemols[imol]->nspecial[iatom], */
            /*                        tag[j]-tagprev); */
            /* else which = 0; */
            if (which == 0) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
              else n++;
            } else if (minimum_image_check(delx,dely,delz)) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
              else n++;
            }
            else if (which > 0) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j ^ (which << SBBITS);
              else n++;
            }
          } else {
            if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
            else n++;
          }
        }
      }
    }
  } else {
    int binxyz[3];
    const int ibin = coord2bin(xtmp, ytmp, ztmp, binxyz);
    const int xbin = binxyz[0];
    const int ybin = binxyz[1];
    const int zbin = binxyz[2];
    for (int k = 0; k < nstencil; k++) {
      const int xbin2 = xbin + stencilxyz(k,0);
      const int ybin2 = ybin + stencilxyz(k,1);
      const int zbin2 = zbin + stencilxyz(k,2);
      if (xbin2 < 0 || xbin2 >= mbinx ||
          ybin2 < 0 || ybin2 >= mbiny ||
          zbin2 < 0 || zbin2 >= mbinz) continue;
      const int jbin = ibin + stencil[k];
      for (int m = 0; m < c_bincount(jbin); m++) {
        const int j = c_bins(jbin,m);

        if (HalfNeigh && j <= i) continue;
        else if (j == i) continue;

        const int jtype = type[j];
        if (exclude && exclusion(i,j,itype,jtype)) continue;

        const X_FLOAT delx = xtmp - x(j,0);
        const X_FLOAT dely = ytmp - x(j,1);
        const X_FLOAT delz = ztmp - x(j,2);
        const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;

        if (rsq <= cutneighsq(itype,jtype)) {
          if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
          else n++;
        }
      }
    }
  }

  neigh_list.d_numneigh(i) = n;

  if (n > neigh_list.maxneighs) {
    resize() = 1;

    if (n > new_maxneighs()) new_maxneighs() = n; // avoid atomics, safe because in while loop
  }
  neigh_list.d_ilist(i) = i;
}

/* ---------------------------------------------------------------------- */

#ifdef LMP_KOKKOS_GPU
template<class DeviceType> template<int HalfNeigh>
LAMMPS_DEVICE_FUNCTION inline
void NeighborKokkosExecute<DeviceType>::build_ItemGhostGPU(typename Kokkos::TeamPolicy<DeviceType>::member_type dev,
                                                      size_t sharedsize) const
{
  auto* sharedmem = static_cast<X_FLOAT *>(dev.team_shmem().get_shmem(sharedsize));

  // loop over atoms in i's bin

  const int atoms_per_bin = c_bins.extent(1);
  const int BINS_PER_TEAM = dev.team_size()/atoms_per_bin <1?1:dev.team_size()/atoms_per_bin;
  const int TEAMS_PER_BIN = atoms_per_bin/dev.team_size()<1?1:atoms_per_bin/dev.team_size();
  const int MY_BIN = dev.team_rank()/atoms_per_bin;

  const int ibin = dev.league_rank()*BINS_PER_TEAM+MY_BIN;

  if (ibin >= mbins) return;

  X_FLOAT* other_x = sharedmem + 5*atoms_per_bin*MY_BIN;
  int* other_id = (int*) &other_x[4 * atoms_per_bin];

  int bincount_current = c_bincount[ibin];

  for (int kk = 0; kk < TEAMS_PER_BIN; kk++) {
    const int MY_II = dev.team_rank()%atoms_per_bin+kk*dev.team_size();
    const int i = MY_II < bincount_current ? c_bins(ibin, MY_II) : -1;

    int n = 0;

    X_FLOAT xtmp;
    X_FLOAT ytmp;
    X_FLOAT ztmp;
    int itype;
    const int index = (i >= 0 && i < nall) ? i : 0;
    const AtomNeighbors neighbors_i = neigh_transpose ?
    neigh_list.get_neighbors_transpose(index) : neigh_list.get_neighbors(index);

    if (i >= 0) {
      xtmp = x(i, 0);
      ytmp = x(i, 1);
      ztmp = x(i, 2);
      itype = type(i);
      other_x[MY_II] = xtmp;
      other_x[MY_II + atoms_per_bin] = ytmp;
      other_x[MY_II + 2 * atoms_per_bin] = ztmp;
      other_x[MY_II + 3 * atoms_per_bin] = itype;
    }
    other_id[MY_II] = i;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int test = (__syncthreads_count(i >= 0 && i < nall) == 0);
    if (test) return;
#elif defined(KOKKOS_ENABLE_SYCL)
    int not_done = (i >= 0 && i < nall);
    dev.team_reduce(Kokkos::Max<int>(not_done));
    if (not_done == 0) return;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
    dev.team_barrier();
#endif

    int which = 0;
    int moltemplate;
    if (molecular == Atom::TEMPLATE) moltemplate = 1;
    else moltemplate = 0;

    const typename ArrayTypes<DeviceType>::t_int_1d_const_um stencil
      = d_stencil;
    const typename ArrayTypes<DeviceType>::t_int_1d_3_const_um stencilxyz
      = d_stencilxyz;

    // loop over all atoms in surrounding bins in stencil including self
    // when i is a ghost atom, must check if stencil bin is out of bounds
    // skip i = j
    // no molecular test when i = ghost atom

    int ghost = (i >= nlocal && i < nall);
    int binxyz[3];
    if (ghost)
      coord2bin(xtmp, ytmp, ztmp, binxyz);
    const int xbin = binxyz[0];
    const int ybin = binxyz[1];
    const int zbin = binxyz[2];
    for (int k = 0; k < nstencil; k++) {
      int active = 1;
      if (ghost) {
        const int xbin2 = xbin + stencilxyz(k,0);
        const int ybin2 = ybin + stencilxyz(k,1);
        const int zbin2 = zbin + stencilxyz(k,2);
        if (xbin2 < 0 || xbin2 >= mbinx ||
            ybin2 < 0 || ybin2 >= mbiny ||
            zbin2 < 0 || zbin2 >= mbinz) active = 0;
      }

      const int jbin = ibin + stencil[k];
      bincount_current = c_bincount[jbin];
      int j = MY_II < bincount_current ? c_bins(jbin, MY_II) : -1;

      if (j >= 0) {
        other_x[MY_II] = x(j, 0);
        other_x[MY_II + atoms_per_bin] = x(j, 1);
        other_x[MY_II + 2 * atoms_per_bin] = x(j, 2);
        other_x[MY_II + 3 * atoms_per_bin] = type(j);
      }

      other_id[MY_II] = j;

      dev.team_barrier();

      if (active && i >= 0 && i < nall) {
        #pragma unroll 4
        for (int m = 0; m < bincount_current; m++) {
          const int j = other_id[m];

          if (HalfNeigh && j <= i) continue;
          else if (j == i) continue;

          const int jtype = other_x[m + 3 * atoms_per_bin];
          if (exclude && exclusion(i,j,itype,jtype)) continue;

          const X_FLOAT delx = xtmp - other_x[m];
          const X_FLOAT dely = ytmp - other_x[m + atoms_per_bin];
          const X_FLOAT delz = ztmp - other_x[m + 2 * atoms_per_bin];
          const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;

          if (rsq <= cutneighsq(itype,jtype)) {
            if (molecular != Atom::ATOMIC && !ghost) {
              if (!moltemplate)
                which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
              /* else if (imol >= 0) */
              /*   which = find_special(onemols[imol]->special[iatom], */
              /*                        onemols[imol]->nspecial[iatom], */
              /*                        tag[j]-tagprev); */
              /* else which = 0; */
              if (which == 0) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
                else n++;
              } else if (minimum_image_check(delx,dely,delz)) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
                else n++;
              }
              else if (which > 0) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = j ^ (which << SBBITS);
                else n++;
              }
            } else {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = j;
              else n++;
            }
          }
        }
      }
      dev.team_barrier();
    }

    if (i >= 0 && i < nall) {
      neigh_list.d_numneigh(i) = n;
      neigh_list.d_ilist(i) = i;
    }

    if (n > neigh_list.maxneighs) {
      resize() = 1;

      if (n > new_maxneighs()) new_maxneighs() = n; // avoid atomics, safe because in while loop
    }
  }
}
#endif

/* ---------------------------------------------------------------------- */

template<class DeviceType> template<int HalfNeigh,int Newton,int Tri>
KOKKOS_FUNCTION
void NeighborKokkosExecute<DeviceType>::
   build_ItemSize(const int &i) const
{
  /* if necessary, goto next page and add pages */
  int n = 0;

  // get subview of neighbors of i

  const AtomNeighbors neighbors_i = neigh_transpose ?
    neigh_list.get_neighbors_transpose(i) : neigh_list.get_neighbors(i);
  const X_FLOAT xtmp = x(i, 0);
  const X_FLOAT ytmp = x(i, 1);
  const X_FLOAT ztmp = x(i, 2);
  const X_FLOAT radi = radius(i);
  const int itype = type(i);
  tagint itag;
  if (HalfNeigh && Newton && Tri) itag = tag(i);

  const int ibin = c_atom2bin(i);

  const typename ArrayTypes<DeviceType>::t_int_1d_const_um stencil
    = d_stencil;

  const int mask_history = 1 << HISTBITS;

  // loop over all bins in neighborhood (includes ibin)
  // loop over rest of atoms in i's bin, ghosts are at end of linked list
  // if j is owned atom, store it, since j is beyond i in linked list
  // if j is ghost, only store if j coords are "above and to the right" of i

  if (HalfNeigh && Newton && !Tri)
  for (int m = 0; m < c_bincount(ibin); m++) {
    const int j = c_bins(ibin,m);

    if (j <= i) continue;
    if (j >= nlocal) {
      if (x(j,2) < ztmp) continue;
      if (x(j,2) == ztmp) {
        if (x(j,1) < ytmp) continue;
        if (x(j,1) == ytmp && x(j,0) < xtmp) continue;
      }
    }

    const int jtype = type(j);
    if (exclude && exclusion(i,j,itype,jtype)) continue;

    const X_FLOAT delx = xtmp - x(j, 0);
    const X_FLOAT dely = ytmp - x(j, 1);
    const X_FLOAT delz = ztmp - x(j, 2);
    const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;
    const X_FLOAT radsum = radi + radius(j);
    const X_FLOAT cutsq = (radsum + skin) * (radsum + skin);

    if (rsq <= cutsq) {
      if (n < neigh_list.maxneighs) {
        int jh = j;
        if (neigh_list.history && rsq < radsum*radsum)
          jh = jh ^ mask_history;

        if (molecular != Atom::ATOMIC) {
          int which = 0;
          if (!moltemplate)
            which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
          /* else if (imol >= 0) */
          /*   which = find_special(onemols[imol]->special[iatom], */
          /*                        onemols[imol]->nspecial[iatom], */
          /*                        tag[j]-tagprev); */
          /* else which = 0; */
          if (which == 0) {
            if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
            else n++;
          } else if (minimum_image_check(delx,dely,delz)) {
            if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
            else n++;
          }
          else if (which > 0) {
            if (n < neigh_list.maxneighs) neighbors_i(n++) = jh ^ (which << SBBITS);
            else n++;
          }
        } else {
          if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
          else n++;
        }
      }
      else n++;
    }
  }

  for (int k = 0; k < nstencil; k++) {
    const int jbin = ibin + stencil[k];

    if (HalfNeigh && Newton && !Tri && (ibin == jbin)) continue;
    // get subview of jbin
    //const ArrayTypes<DeviceType>::t_int_1d_const_um =Kokkos::subview<t_int_1d_const_um>(bins,jbin,ALL);
    for (int m = 0; m < c_bincount(jbin); m++) {

      const int j = c_bins(jbin,m);

      if (HalfNeigh && !Newton && j <= i) continue;
      if (!HalfNeigh && j == i) continue;

      // for triclinic, bin stencil is full in all 3 dims
      // must use itag/jtag to eliminate half the I/J interactions
      // cannot use I/J exact coord comparision
      //   b/c transforming orthog -> lambda -> orthog for ghost atoms
      //   with an added PBC offset can shift all 3 coords by epsilon

      if (HalfNeigh && Newton && Tri) {
        if (j <= i) continue;
        if (j >= nlocal) {
          const tagint jtag = tag(j);
          if (itag > jtag) {
            if ((itag+jtag) % 2 == 0) continue;
          } else if (itag < jtag) {
            if ((itag+jtag) % 2 == 1) continue;
          } else {
            if (fabs(x(j,2)-ztmp) > delta) {
              if (x(j,2) < ztmp) continue;
            } else if (fabs(x(j,1)-ytmp) > delta) {
              if (x(j,1) < ytmp) continue;
            } else {
              if (x(j,0) < xtmp) continue;
            }
          }
        }
      }

      const int jtype = type(j);
      if (exclude && exclusion(i,j,itype,jtype)) continue;

      const X_FLOAT delx = xtmp - x(j, 0);
      const X_FLOAT dely = ytmp - x(j, 1);
      const X_FLOAT delz = ztmp - x(j, 2);
      const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;
      const X_FLOAT radsum = radi + radius(j);
      const X_FLOAT cutsq = (radsum + skin) * (radsum + skin);

      if (rsq <= cutsq) {
        if (n < neigh_list.maxneighs) {

          int jh = j;
          if (neigh_list.history && rsq < radsum*radsum)
            jh = jh ^ mask_history;

          if (molecular != Atom::ATOMIC) {
            int which = 0;
            if (!moltemplate)
              which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
            /* else if (imol >= 0) */
            /*   which = find_special(onemols[imol]->special[iatom], */
            /*                        onemols[imol]->nspecial[iatom], */
            /*                        tag[j]-tagprev); */
            /* else which = 0; */
            if (which == 0) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
              else n++;
            } else if (minimum_image_check(delx,dely,delz)) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
              else n++;
            }
            else if (which > 0) {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = jh ^ (which << SBBITS);
              else n++;
            }
          } else {
            if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
            else n++;
          }
        }
        else n++;
      }
    }
  }

  neigh_list.d_numneigh(i) = n;

  if (n > neigh_list.maxneighs) {
    resize() = 1;

    if (n > new_maxneighs()) new_maxneighs() = n; // avoid atomics, safe because in while loop
  }

  neigh_list.d_ilist(i) = i;
}

/* ---------------------------------------------------------------------- */

#ifdef LMP_KOKKOS_GPU
template<class DeviceType> template<int HalfNeigh,int Newton,int Tri>
LAMMPS_DEVICE_FUNCTION inline
void NeighborKokkosExecute<DeviceType>::build_ItemSizeGPU(typename Kokkos::TeamPolicy<DeviceType>::member_type dev,
                                                          size_t sharedsize) const
{
  auto* sharedmem = static_cast<X_FLOAT *>(dev.team_shmem().get_shmem(sharedsize));

  // loop over atoms in i's bin

  const int atoms_per_bin = c_bins.extent(1);
  const int BINS_PER_TEAM = dev.team_size()/atoms_per_bin <1?1:dev.team_size()/atoms_per_bin;
  const int TEAMS_PER_BIN = atoms_per_bin/dev.team_size()<1?1:atoms_per_bin/dev.team_size();
  const int MY_BIN = dev.team_rank()/atoms_per_bin;

  const int ibin = dev.league_rank()*BINS_PER_TEAM+MY_BIN;

  if (ibin >= mbins) return;

  X_FLOAT* other_x = sharedmem + 7*atoms_per_bin*MY_BIN;
  int* other_id = (int*) &other_x[6 * atoms_per_bin];

  int bincount_current = c_bincount[ibin];

  for (int kk = 0; kk < TEAMS_PER_BIN; kk++) {
    const int MY_II = dev.team_rank()%atoms_per_bin+kk*dev.team_size();
    const int i = MY_II < bincount_current ? c_bins(ibin, MY_II) : -1;

    int n = 0;

    X_FLOAT xtmp;
    X_FLOAT ytmp;
    X_FLOAT ztmp;
    X_FLOAT radi;
    int itype;
    tagint itag;
    const int index = (i >= 0 && i < nlocal) ? i : 0;
    const AtomNeighbors neighbors_i = neigh_transpose ?
    neigh_list.get_neighbors_transpose(index) : neigh_list.get_neighbors(index);
    const int mask_history = 1 << HISTBITS;

    if (i >= 0) {
      xtmp = x(i, 0);
      ytmp = x(i, 1);
      ztmp = x(i, 2);
      radi = radius(i);
      itype = type(i);
      other_x[MY_II] = xtmp;
      other_x[MY_II + atoms_per_bin] = ytmp;
      other_x[MY_II + 2 * atoms_per_bin] = ztmp;
      other_x[MY_II + 3 * atoms_per_bin] = itype;
      other_x[MY_II + 4 * atoms_per_bin] = radi;
      if (HalfNeigh && Newton && Tri) {
        itag = tag(i);
        other_x[MY_II + 5 * atoms_per_bin] = itag;
      }
    }
    other_id[MY_II] = i;
#if defined(KOKKOS_ENABLE_CUDA) || defined(KOKKOS_ENABLE_HIP)
    int test = (__syncthreads_count(i >= 0 && i < nlocal) == 0);
    if (test) return;
#elif defined(KOKKOS_ENABLE_SYCL)
    int not_done = (i >= 0 && i < nlocal);
    dev.team_reduce(Kokkos::Max<int>(not_done));
    if (not_done == 0) return;
#elif defined(KOKKOS_ENABLE_OPENMPTARGET)
    dev.team_barrier();
#endif

    if (HalfNeigh && Newton && !Tri)
    if (i >= 0 && i < nlocal) {
      #pragma unroll 4
      for (int m = 0; m < bincount_current; m++) {
        int j = other_id[m];

        if (j <= i) continue;
        if (j >= nlocal) {
          if (x(j,2) < ztmp) continue;
          if (x(j,2) == ztmp) {
            if (x(j,1) < ytmp) continue;
            if (x(j,1) == ytmp && x(j,0) < xtmp) continue;
          }
        }

        const int jtype = other_x[m + 3 * atoms_per_bin];
        if (exclude && exclusion(i,j,itype,jtype)) continue;
        const X_FLOAT delx = xtmp - other_x[m];
        const X_FLOAT dely = ytmp - other_x[m + atoms_per_bin];
        const X_FLOAT delz = ztmp - other_x[m + 2 * atoms_per_bin];
        const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;
        const X_FLOAT radsum = radi + other_x[m + 4 * atoms_per_bin];
        const X_FLOAT cutsq = (radsum + skin) * (radsum + skin);

        if (rsq <= cutsq) {
          if (n < neigh_list.maxneighs) {

            int jh = j;
            if (neigh_list.history && rsq < radsum*radsum)
              jh = jh ^ mask_history;

            if (molecular != Atom::ATOMIC) {
              int which = 0;
              if (!moltemplate)
                which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
              /* else if (imol >= 0) */
              /*   which = find_special(onemols[imol]->special[iatom], */
              /*                        onemols[imol]->nspecial[iatom], */
              /*                        tag[j]-tagprev); */
              /* else which = 0; */
              if (which == 0) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
                else n++;
              } else if (minimum_image_check(delx,dely,delz)) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
                else n++;
              }
              else if (which > 0) {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = jh ^ (which << SBBITS);
                else n++;
              }
            } else {
              if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
              else n++;
            }
          }
          else n++;
        }
      }
    }
    dev.team_barrier();

    const typename ArrayTypes<DeviceType>::t_int_1d_const_um stencil
      = d_stencil;
    for (int k = 0; k < nstencil; k++) {
      const int jbin = ibin + stencil[k];

      if (ibin == jbin) continue;
      if (HalfNeigh && Newton && !Tri && (ibin == jbin)) continue;

      bincount_current = c_bincount[jbin];
      int j = MY_II < bincount_current ? c_bins(jbin, MY_II) : -1;

      if (j >= 0) {
        other_x[MY_II] = x(j, 0);
        other_x[MY_II + atoms_per_bin] = x(j, 1);
        other_x[MY_II + 2 * atoms_per_bin] = x(j, 2);
        other_x[MY_II + 3 * atoms_per_bin] = type(j);
        other_x[MY_II + 4 * atoms_per_bin] = radius(j);
        if (HalfNeigh && Newton && Tri)
          other_x[MY_II + 5 * atoms_per_bin] = tag(j);
      }

      other_id[MY_II] = j;

      dev.team_barrier();

      if (i >= 0 && i < nlocal) {
        #pragma unroll 8
        for (int m = 0; m < bincount_current; m++) {
          const int j = other_id[m];

          if (HalfNeigh && !Newton && j <= i) continue;
          if (!HalfNeigh && j == i) continue;

          // for triclinic, bin stencil is full in all 3 dims
          // must use itag/jtag to eliminate half the I/J interactions
          // cannot use I/J exact coord comparision
          //   b/c transforming orthog -> lambda -> orthog for ghost atoms
          //   with an added PBC offset can shift all 3 coords by epsilon

          if (HalfNeigh && Newton && Tri) {
            if (j <= i) continue;
            if (j >= nlocal) {
              const tagint jtag = other_x[m + 5 * atoms_per_bin];
              if (itag > jtag) {
                if ((itag+jtag) % 2 == 0) continue;
              } else if (itag < jtag) {
                if ((itag+jtag) % 2 == 1) continue;
              } else {
                if (fabs(x(j,2)-ztmp) > delta) {
                  if (x(j,2) < ztmp) continue;
                } else if (fabs(x(j,1)-ytmp) > delta) {
                  if (x(j,1) < ytmp) continue;
                } else {
                  if (x(j,0) < xtmp) continue;
                }
              }
            }
          }

          const int jtype = other_x[m + 3 * atoms_per_bin];
          if (exclude && exclusion(i,j,itype,jtype)) continue;

          const X_FLOAT delx = xtmp - other_x[m];
          const X_FLOAT dely = ytmp - other_x[m + atoms_per_bin];
          const X_FLOAT delz = ztmp - other_x[m + 2 * atoms_per_bin];
          const X_FLOAT rsq = delx*delx + dely*dely + delz*delz;
          const X_FLOAT radsum = radi + other_x[m + 4 * atoms_per_bin];
          const X_FLOAT cutsq = (radsum + skin) * (radsum + skin);

          if (rsq <= cutsq) {
            if (n < neigh_list.maxneighs) {

              int jh = j;
              if (neigh_list.history && rsq < radsum*radsum)
                jh = jh ^ mask_history;

              if (molecular != Atom::ATOMIC) {
                int which = 0;
                if (!moltemplate)
                  which = NeighborKokkosExecute<DeviceType>::find_special(i,j);
                /* else if (imol >= 0) */
                /*   which = find_special(onemols[imol]->special[iatom], */
                /*                        onemols[imol]->nspecial[iatom], */
                /*                        tag[j]-tagprev); */
                /* else which = 0; */
                if (which == 0) {
                  if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
                  else n++;
                } else if (minimum_image_check(delx,dely,delz)) {
                  if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
                  else n++;
                }
                else if (which > 0) {
                  if (n < neigh_list.maxneighs) neighbors_i(n++) = jh ^ (which << SBBITS);
                  else n++;
                }
              } else {
                if (n < neigh_list.maxneighs) neighbors_i(n++) = jh;
                else n++;
              }
            }
            else n++;
          }
        }
      }
      dev.team_barrier();
    }

    if (i >= 0 && i < nlocal) {
      neigh_list.d_numneigh(i) = n;
      neigh_list.d_ilist(i) = i;
    }

    if (n > neigh_list.maxneighs) {
      resize() = 1;

      if (n > new_maxneighs()) new_maxneighs() = n; // avoid atomics, safe because in while loop
    }
  }
}
#endif

}

namespace LAMMPS_NS {
template class NPairKokkos<LMPDeviceType,0,0,0,0,0>;
template class NPairKokkos<LMPDeviceType,0,0,1,0,0>;
template class NPairKokkos<LMPDeviceType,1,1,0,0,0>;
template class NPairKokkos<LMPDeviceType,1,0,0,0,0>;
template class NPairKokkos<LMPDeviceType,1,1,1,0,0>;
template class NPairKokkos<LMPDeviceType,1,0,1,0,0>;
template class NPairKokkos<LMPDeviceType,1,1,0,1,0>;
template class NPairKokkos<LMPDeviceType,1,0,0,1,0>;
template class NPairKokkos<LMPDeviceType,1,1,0,0,1>;
template class NPairKokkos<LMPDeviceType,1,0,0,0,1>;
template class NPairKokkos<LMPDeviceType,1,1,0,1,1>;
template class NPairKokkos<LMPDeviceType,1,0,0,1,1>;
#ifdef LMP_KOKKOS_GPU
template class NPairKokkos<LMPHostType,0,0,0,0,0>;
template class NPairKokkos<LMPHostType,0,0,1,0,0>;
template class NPairKokkos<LMPHostType,1,1,0,0,0>;
template class NPairKokkos<LMPHostType,1,0,0,0,0>;
template class NPairKokkos<LMPHostType,1,1,1,0,0>;
template class NPairKokkos<LMPHostType,1,0,1,0,0>;
template class NPairKokkos<LMPHostType,1,1,0,1,0>;
template class NPairKokkos<LMPHostType,1,0,0,1,0>;
template class NPairKokkos<LMPHostType,1,1,0,0,1>;
template class NPairKokkos<LMPHostType,1,0,0,0,1>;
template class NPairKokkos<LMPHostType,1,1,0,1,1>;
template class NPairKokkos<LMPHostType,1,0,0,1,1>;
#endif
}
