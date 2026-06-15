#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cmath>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

#include "attention_tiled_cuda.h"

namespace {


template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void attention_tiled_online_softmax_kernel_stub_v3(
  const __nv_bfloat16* __restrict__ query_ptr,
  const __nv_bfloat16* __restrict__ key_ptr,
  const __nv_bfloat16* __restrict__ value_ptr,
  __nv_bfloat16* __restrict__ output_ptr,

  int batch_size,
  int num_heads,
  int q_seq_len,
  int kv_seq_len,

  int q_stride_b,
  int q_stride_h,
  int q_stride_n,

  int k_stride_b,
  int k_stride_h,
  int k_stride_n,

  int v_stride_b,
  int v_stride_h,
  int v_stride_n,

  int o_stride_b,
  int o_stride_h,
  int o_stride_n,

  float scale,
  bool causal
) {
  using namespace nvcuda;

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 16;
  constexpr int WMMA_K = 16;

  static_assert(BLOCK_M % WMMA_M == 0, "BLOCK_M must be a multiple of 16");
  static_assert(BLOCK_N % WMMA_N == 0, "BLOCK_N must be a multiple of 16");
  static_assert(HEAD_DIM % WMMA_K == 0, "HEAD_DIM must be a multiple of 16");

  __shared__ __nv_bfloat16 q_smem[BLOCK_M][HEAD_DIM];
  __shared__ __nv_bfloat16 k_smem[BLOCK_N][HEAD_DIM];
  __shared__ __nv_bfloat16 v_smem[BLOCK_N][HEAD_DIM];
  __shared__ __nv_bfloat16 p_smem[BLOCK_M][BLOCK_N];

  __shared__ float s_smem[BLOCK_M][BLOCK_N];
  __shared__ float o_smem[BLOCK_M][HEAD_DIM];

  __shared__ float l_smem[BLOCK_M];
  __shared__ float m_smem[BLOCK_M];

  int batch_id = blockIdx.z;
  int head_id = blockIdx.y;
  int tile_id = blockIdx.x;

  const __nv_bfloat16* q_tile = query_ptr +
               batch_id * q_stride_b +
               head_id * q_stride_h +
               tile_id * BLOCK_M * q_stride_n;

  const __nv_bfloat16* k_tile = key_ptr +
               batch_id * k_stride_b +
               head_id * k_stride_h;

  const __nv_bfloat16* v_tile = value_ptr +
               batch_id * v_stride_b +
               head_id * v_stride_h;

  __nv_bfloat16* o_tile = output_ptr +
               batch_id * o_stride_b +
               head_id * o_stride_h +
               tile_id * BLOCK_M * o_stride_n;

  // Load Q tile once and init output accumulator.
  for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
    int row = i / HEAD_DIM;
    int col = i % HEAD_DIM;
    int global_index_row = tile_id * BLOCK_M + row;
    q_smem[row][col] = (global_index_row < q_seq_len) ? q_tile[row * q_stride_n + col] : __float2bfloat16(0.0f);
  }

  for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
    int row = i / HEAD_DIM;
    int col = i % HEAD_DIM;
    o_smem[row][col] = 0.0f;
  }

  for (int row = threadIdx.x; row < BLOCK_M; row += blockDim.x) {
    l_smem[row] = 0.0f;
    m_smem[row] = -INFINITY;
  }
  __syncthreads();

  // Each warp computes one 16x16 output sub-tile.
  constexpr int NUM_WARPS_M = BLOCK_M / WMMA_M;
  constexpr int NUM_WARPS_N = BLOCK_N / WMMA_N;
  constexpr int NUM_WARPS = NUM_WARPS_M * NUM_WARPS_N;
  constexpr int WARP_SIZE = 32;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int warp_m = warp_id / NUM_WARPS_N;
  const int warp_n = warp_id % NUM_WARPS_N;

  const int iter_loops = (kv_seq_len + BLOCK_N - 1) / BLOCK_N;
  for (int kv_tile_id = 0; kv_tile_id < iter_loops; ++kv_tile_id) {
    int kv_start_row = BLOCK_N * kv_tile_id;

    // Load K tile into shared memory.
    for (int i = threadIdx.x; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
      int row = i / HEAD_DIM;
      int col = i % HEAD_DIM;
      int start_tile = kv_start_row + row;
      k_smem[row][col] = (start_tile < kv_seq_len) ? k_tile[start_tile * k_stride_n + col] : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // ---- Tensor Core GEMM: S = Q @ K^T ----
    // A = Q      : BLOCK_M x HEAD_DIM, row-major
    // B = K^T    : HEAD_DIM x BLOCK_N, col-major (k_smem is physically row-major
    //              over K, which is exactly the col-major layout of K^T).
    // C = S      : BLOCK_M x BLOCK_N, row-major
    if (warp_id < NUM_WARPS) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_q;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::col_major> frag_kt;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_s;
      wmma::fill_fragment(frag_s, 0.0f);

      #pragma unroll
      for (int k_step = 0; k_step < HEAD_DIM / WMMA_K; ++k_step) {
        wmma::load_matrix_sync(frag_q, &q_smem[warp_m * WMMA_M][k_step * WMMA_K], HEAD_DIM);
        wmma::load_matrix_sync(frag_kt, &k_smem[warp_n * WMMA_N][k_step * WMMA_K], HEAD_DIM);
        wmma::mma_sync(frag_s, frag_q, frag_kt, frag_s);
      }
      wmma::store_matrix_sync(&s_smem[warp_m * WMMA_M][warp_n * WMMA_N], frag_s, BLOCK_N, wmma::mem_row_major);
    }
    __syncthreads();

    // Apply scale and causal mask.
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
      int row = idx / BLOCK_N;
      int col = idx % BLOCK_N;
      float s_ij = s_smem[row][col] * scale;

      int global_q_row = tile_id * BLOCK_M + row;
      int global_kv_row = kv_start_row + col;
      if (causal && global_kv_row > global_q_row) s_ij = -INFINITY;
      if (global_q_row >= q_seq_len || global_kv_row >= kv_seq_len) s_ij = -INFINITY;
      s_smem[row][col] = s_ij;
    }
    __syncthreads();

    // Online softmax in FP32.
    if (threadIdx.x < BLOCK_M) {
      int row = threadIdx.x;
      float m_tile = -INFINITY;
      for (int col = 0; col < BLOCK_N; ++col) {
        m_tile = fmaxf(m_tile, s_smem[row][col]);
      }

      float max_prev = m_smem[row];
      float max_new = fmaxf(max_prev, m_tile);

      float scale_old = expf(max_prev - max_new);
      for (int d = 0; d < HEAD_DIM; ++d) {
        o_smem[row][d] *= scale_old;
      }
      l_smem[row] *= scale_old;

      float l_tile = 0.0f;
      for (int col = 0; col < BLOCK_N; ++col) {
        float p_ij = expf(s_smem[row][col] - max_new);
        l_tile += (max_new == -INFINITY) ? 0.0f : p_ij;
        s_smem[row][col] = (max_new == -INFINITY) ? 0.0f : p_ij;
      }

      l_smem[row] += l_tile;
      m_smem[row] = max_new;
    }
    __syncthreads();

    // Convert P to bfloat16 for the next Tensor Core GEMM.
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
      int row = idx / BLOCK_N;
      int col = idx % BLOCK_N;
      p_smem[row][col] = __float2bfloat16(s_smem[row][col]);
    }

    // Load V tile into shared memory.
    for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
      int row = idx / HEAD_DIM;
      int col = idx % HEAD_DIM;
      int start_tile = kv_start_row + row;
      v_smem[row][col] = (start_tile < kv_seq_len)
                          ? v_tile[start_tile * v_stride_n + col]
                          : __float2bfloat16(0.0f);
    }
    __syncthreads();

    // ---- Tensor Core GEMM: O += P @ V ----
    // A = P : BLOCK_M x BLOCK_N, row-major
    // B = V : BLOCK_N x HEAD_DIM, row-major
    // C = O : BLOCK_M x HEAD_DIM, row-major
    if (warp_id < NUM_WARPS) {
      wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_p;
      wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> frag_v;
      wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_o;

      // Start from the running FP32 output accumulator.
      wmma::load_matrix_sync(frag_o, &o_smem[warp_m * WMMA_M][warp_n * WMMA_N], HEAD_DIM, wmma::mem_row_major);

      #pragma unroll
      for (int k_step = 0; k_step < BLOCK_N / WMMA_K; ++k_step) {
        wmma::load_matrix_sync(frag_p, &p_smem[warp_m * WMMA_M][k_step * WMMA_K], BLOCK_N);
        wmma::load_matrix_sync(frag_v, &v_smem[k_step * WMMA_K][warp_n * WMMA_N], HEAD_DIM);
        wmma::mma_sync(frag_o, frag_p, frag_v, frag_o);
      }
      wmma::store_matrix_sync(&o_smem[warp_m * WMMA_M][warp_n * WMMA_N], frag_o, HEAD_DIM, wmma::mem_row_major);
    }
    __syncthreads();
  }

  // Finalize and write back O.
  for (int idx = threadIdx.x; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
    int row = idx / HEAD_DIM;
    int col = idx % HEAD_DIM;
    int global_q_row = tile_id * BLOCK_M + row;
    if (global_q_row >= q_seq_len) continue;
    float o = (l_smem[row] > 0.0f) ? (o_smem[row][col] / l_smem[row]) : 0.0f;
    o_tile[row * o_stride_n + col] = __float2bfloat16(o);
  }
}

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
void launch_attention_v3(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    torch::Tensor& output_tensor,
    int batch_size,
    int num_heads,
    int q_seq_len,
    int kv_seq_len,
    float softmax_scale,
    bool causal,
    dim3 grid,
    dim3 block) {
  auto query_ptr = reinterpret_cast<const __nv_bfloat16*>(query.data_ptr());
  auto key_ptr = reinterpret_cast<const __nv_bfloat16*>(key.data_ptr());
  auto value_ptr = reinterpret_cast<const __nv_bfloat16*>(value.data_ptr());
  auto output_ptr = reinterpret_cast<__nv_bfloat16*>(output_tensor.data_ptr());

  attention_tiled_online_softmax_kernel_stub_v3<BLOCK_M, BLOCK_N, HEAD_DIM><<<grid, block>>>(
      query_ptr,
      key_ptr,
      value_ptr,
      output_ptr,
      batch_size,
      num_heads,
      q_seq_len,
      kv_seq_len,
      static_cast<int>(query.stride(0)),
      static_cast<int>(query.stride(1)),
      static_cast<int>(query.stride(2)),
      static_cast<int>(key.stride(0)),
      static_cast<int>(key.stride(1)),
      static_cast<int>(key.stride(2)),
      static_cast<int>(value.stride(0)),
      static_cast<int>(value.stride(1)),
      static_cast<int>(value.stride(2)),
      static_cast<int>(output_tensor.stride(0)),
      static_cast<int>(output_tensor.stride(1)),
      static_cast<int>(output_tensor.stride(2)),
      softmax_scale,
      causal
  );
}

}  // namespace

torch::Tensor attention_tiled_online_softmax_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal,
    c10::optional<double> scale,
    int64_t tile_size) {
  (void)tile_size;

  TORCH_CHECK(query.is_cuda(), "query must be a CUDA tensor");
  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");

  TORCH_CHECK(
      query.scalar_type() == torch::kBFloat16,
      "query must be a BFloat16 dtype tensor");
  TORCH_CHECK(
      key.scalar_type() == torch::kBFloat16,
      "key must be a BFloat16 dtype tensor");
  TORCH_CHECK(
      value.scalar_type() == torch::kBFloat16,
      "value must be a BFloat16 dtype tensor");

  const int batch_size = static_cast<int>(query.size(0));
  const int num_heads = static_cast<int>(query.size(1));
  const int q_seq_len = static_cast<int>(query.size(2));
  const int value_dim = static_cast<int>(value.size(3));
  const int k_seq_len = static_cast<int>(key.size(2));
  // :TODO add assert for head_dim
  const float softmax_scale = scale.has_value()
      ? static_cast<float>(*scale)
      : 1.0f / std::sqrt(static_cast<float>(query.size(3)));

  auto output_tensor = torch::zeros(
    {batch_size, num_heads, q_seq_len, value_dim},
    query.options()
  );
  auto l_tensor = torch::zeros(
      {batch_size, num_heads, q_seq_len},
      query.options().dtype(torch::kFloat32));

  TORCH_CHECK(l_tensor.is_cuda());
  TORCH_CHECK(l_tensor.is_contiguous());
  TORCH_CHECK(l_tensor.dtype() == torch::kFloat32);

  auto m_tensor = torch::full(
      {batch_size, num_heads, q_seq_len},
      -std::numeric_limits<float>::infinity(),
      query.options().dtype(torch::kFloat32));

  TORCH_CHECK(m_tensor.is_cuda());
  TORCH_CHECK(m_tensor.is_contiguous());
  TORCH_CHECK(m_tensor.dtype() == torch::kFloat32);

  constexpr int kBlockM = 32;
  constexpr int kBlockN = 64;
  constexpr int kHeadDim = 64;
  TORCH_CHECK(query.size(3) == kHeadDim, "expected head_dim == 64");
  const int num_m_block = (q_seq_len + kBlockM - 1) / kBlockM;

  dim3 grid(num_m_block, num_heads, batch_size);
  dim3 block(256);

  launch_attention_v3<kBlockM, kBlockN, kHeadDim>(
      query,
      key,
      value,
      output_tensor,
      batch_size,
      num_heads,
      q_seq_len,
      k_seq_len,
      softmax_scale,
      causal,
      grid,
      block);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output_tensor;
}
