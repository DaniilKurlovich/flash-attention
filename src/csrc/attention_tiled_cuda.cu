#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>

#include <cmath>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

#include "attention_tiled_cuda.h"

namespace {

constexpr int BLOCK_DIM = 256;

template <int ROW_SIZE, int COL_SIZE, int BLOCK_SIZE>
__device__ inline void global_to_shared_copy_by_tile_in_bf16(const nv_bfloat16* src, uint32_t dst,
                                                             const int seq_len, const int global_tile_row, const int global_stride) {
  constexpr int num_elems_copy_in_nv_bfloat16 = 16 / sizeof(nv_bfloat16);
  const int total_chunks = ROW_SIZE * COL_SIZE / num_elems_copy_in_nv_bfloat16;
  const int tid = threadIdx.x;

  for (int i = tid; i < total_chunks; i += BLOCK_SIZE) {
      int idx = i * num_elems_copy_in_nv_bfloat16;
      int row = idx / COL_SIZE;
      int col = idx % COL_SIZE;

      const int global_row = global_tile_row + row;
      if (global_row >= seq_len) {
          continue;
      }
      const nv_bfloat16 *src_addr = src + global_row * global_stride + col;
      const uint32_t dst_addr = dst + (row * COL_SIZE + col) * sizeof(nv_bfloat16);
      asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(dst_addr), "l"(src_addr));
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_all;");
  __syncthreads();
}

__device__ inline void ldmatrix_x4(uint32_t dst[4], uint32_t src) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3])
        : "r"(src)
    );
}

__device__ inline void ldmatrix_x2(uint32_t dst[2], uint32_t src) {
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.b16 {%0, %1}, [%2];"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(src)
    );
}

__device__ inline void ldmatrix_x2_transpose(uint32_t dst[2], uint32_t src) {
    asm volatile (
        "ldmatrix.sync.aligned.m8n8.x2.trans.b16 {%0, %1}, [%2];"
        : "=r"(dst[0]), "=r"(dst[1])
        : "r"(src)
    );
}

// computes matrix multiplication C = A @ B^T
__device__ inline void mma_m16n8k16(uint32_t a[4], uint32_t b[2], float c[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

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
  // using namespace nvcuda;

  constexpr int WMMA_M = 16;
  constexpr int WMMA_N = 8;
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

  int batch_id = blockIdx.z;
  int head_id = blockIdx.y;
  int tile_id = blockIdx.x;
  const int tid = threadIdx.x;

  const __nv_bfloat16* q_tile = query_ptr +
               batch_id * q_stride_b +
               head_id * q_stride_h;

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

  uint32_t Q_rmem[BLOCK_M / WMMA_M][HEAD_DIM / WMMA_K][4] = {};
  uint32_t K_rmem[BLOCK_N / WMMA_N][HEAD_DIM / WMMA_K][2] = {};
  uint32_t V_rmem[BLOCK_N / WMMA_N][HEAD_DIM / WMMA_K][2] = {};
  uint32_t P_rmem[BLOCK_M / WMMA_M][BLOCK_N / WMMA_N][4] = {};

  float O_rmem[BLOCK_M / WMMA_M][HEAD_DIM / WMMA_K][4] = {};

  float max_prev[BLOCK_M / WMMA_M][2] = {};    // in .m16n8k16 accumulator per thread store 4 register, therefore for reduction on row need only 2
  float row_sumexp[BLOCK_M / WMMA_M][2] = {};

  // init
  for (int mma_row_id = 0; mma_row_id < BLOCK_M / WMMA_M; ++mma_row_id) {
      max_prev[mma_row_id][0] = -INFINITY;
      max_prev[mma_row_id][1] = -INFINITY;
  }

  uint32_t q_addr_smem = __cvta_generic_to_shared(q_smem);
  uint32_t k_addr_smem = __cvta_generic_to_shared(k_smem);
  uint32_t v_addr_smem = __cvta_generic_to_shared(v_smem);
  int global_start_row = tile_id * BLOCK_M;
  global_to_shared_copy_by_tile_in_bf16<BLOCK_M, HEAD_DIM, BLOCK_DIM>(q_tile, q_addr_smem, q_seq_len, global_start_row, q_stride_n);

  const int warp_id = threadIdx.x / 32;
  const int lane_id = threadIdx.x % 32;

  for (int mma_row_id = 0; mma_row_id < BLOCK_M / WMMA_M; ++mma_row_id) {
      for (int mma_col_id = 0; mma_col_id < HEAD_DIM / WMMA_K; ++mma_col_id) {
          // lanes 0-15: row offset 0..15, col offset 0
          // lanes 16-31: row offset 0..15, col offset 8 (NOTE: this split its need for ldmatrix 8x8.x4 convention)
          const int row = mma_row_id * WMMA_N + (lane_id % 16);
          const int col = mma_col_id * WMMA_K + (lane_id / 16) * 8;
          uint32_t q_addr_smem_src = q_addr_smem + (row * BLOCK_M + col) * sizeof(half);
          ldmatrix_x4(Q_rmem[mma_row_id][mma_col_id], q_addr_smem_src);
      }
  }

  const int iter_loops = (kv_seq_len + BLOCK_N - 1) / BLOCK_N;
  for (int kv_tile_id = 0; kv_tile_id < iter_loops; ++kv_tile_id) {
    int kv_start_row = BLOCK_N * kv_tile_id;

    // Load K tile into shared memory.
    global_to_shared_copy_by_tile_in_bf16<BLOCK_N, HEAD_DIM, BLOCK_DIM>(k_tile, k_addr_smem, kv_seq_len, kv_start_row, k_stride_n);

    // Load K tile into registers.
    for (int mma_row_id = 0; mma_row_id < BLOCK_N / WMMA_N; ++mma_row_id) {
        for (int mma_col_id = 0; mma_col_id < HEAD_DIM / WMMA_K; ++mma_col_id) {
            // lanes 0-7: row offset 0..8;
            // lanes 8-15: row offset 0..8;
            const int row = mma_row_id * WMMA_N + (lane_id % 8);
            // lanes 0-7: col offset 0;
            // lanes 8-16: col offset 8
            const int col = mma_col_id * WMMA_K + (lane_id / 8) * 8;
            uint32_t k_addr_smem_src = k_addr_smem + (row * BLOCK_N + col) * sizeof(half);
            ldmatrix_x2(K_rmem[mma_row_id][mma_col_id], k_addr_smem_src);
        }
    }

    // ---- Tensor Core GEMM: S = Q @ K^T ----
    // A = Q      : BLOCK_M x HEAD_DIM, row-major
    // B = K^T    : HEAD_DIM x BLOCK_N, col-major (k_smem is physically row-major
    //              over K, which is exactly the col-major layout of K^T).
    // C = S      : BLOCK_M x BLOCK_N, row-major
    float S_rmem[BLOCK_M / WMMA_M][BLOCK_N / WMMA_N][4] = {};
    for (int mma_q_id = 0; mma_q_id < BLOCK_M / WMMA_M; ++mma_q_id) {
        for (int mma_k_id = 0; mma_k_id < BLOCK_N / WMMA_N; ++mma_k_id) {
            for (int mma_tile_id = 0; mma_tile_id < HEAD_DIM / WMMA_K; ++mma_tile_id) {
                mma_m16n8k16(Q_rmem[mma_q_id][mma_tile_id], K_rmem[mma_k_id][mma_tile_id], S_rmem[mma_q_id][mma_k_id]);
            }
        }
    }

    // apply scale and compute online softmax
    for (int mma_row_id = 0; mma_row_id < BLOCK_M / WMMA_M; ++mma_row_id) {
        for (int mma_col_id = 0; mma_col_id < BLOCK_N / WMMA_N; ++mma_col_id) {
            for (int d = 0; d < 4; ++d) {
                S_rmem[mma_row_id][mma_col_id][d] *= scale;
            }
        }

        float cur_rowmax[2] = {-INFINITY, -INFINITY};
        for (int mma_col_id = 0; mma_col_id < BLOCK_N / WMMA_N; ++mma_col_id) {
            float* req = S_rmem[mma_row_id][mma_col_id];
            cur_rowmax[0] = max(cur_rowmax[0], max(req[0], req[1]));    // top 8 rows
            cur_rowmax[1] = max(cur_rowmax[1], max(req[2], req[3]));    // bottom 8  rows
        }

        // reduce neighbour threads from same row
        // https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-fragment-mma-16816-float:~:text=Figure%2078%20MMA,%EF%83%81
        cur_rowmax[0] = max(__shfl_xor_sync(0xffffffff, cur_rowmax[0], 1), cur_rowmax[0]);  // t0 <> t1, t2 <> t3
        cur_rowmax[0] = max(__shfl_xor_sync(0xffffffff, cur_rowmax[0], 2), cur_rowmax[0]);  // t0 <> t2, t1 <> t3
        cur_rowmax[1] = max(__shfl_xor_sync(0xffffffff, cur_rowmax[1], 1), cur_rowmax[1]);
        cur_rowmax[1] = max(__shfl_xor_sync(0xffffffff, cur_rowmax[1], 2), cur_rowmax[1]);

        // rescale
        float scale[2] = {expf(max_prev[mma_row_id][0] - cur_rowmax[0]), expf(max_prev[mma_row_id][1] - cur_rowmax[1])};
        // [BLOCK_M / WMMA_M][HEAD_DIM / WMMA_K]
        for (int mma_col_id = 0; mma_col_id < HEAD_DIM / WMMA_K; ++mma_col_id) {
            O_rmem[mma_row_id][mma_col_id][0] *= scale[0];
            O_rmem[mma_row_id][mma_col_id][1] *= scale[0];
            O_rmem[mma_row_id][mma_col_id][2] *= scale[1];
            O_rmem[mma_row_id][mma_col_id][3] *= scale[1];
        }

        // simply write back with no condition
        max_prev[mma_row_id][0] = cur_rowmax[0];
        max_prev[mma_row_id][1] = cur_rowmax[1];

        // compute P and rowsumexp
        float cur_rowsumexp[2] = {};
        for (int mma_col_id = 0; mma_col_id < BLOCK_N / WMMA_N; ++mma_col_id) {
            // inplace modify matrix S
            float* reqs = S_rmem[mma_row_id][mma_col_id];
            reqs[0] = __expf(S_rmem[mma_row_id][mma_col_id][0] - max_prev[mma_row_id][0]);
            reqs[1] = __expf(S_rmem[mma_row_id][mma_col_id][1] - max_prev[mma_row_id][0]);
            reqs[2] = __expf(S_rmem[mma_row_id][mma_col_id][2] - max_prev[mma_row_id][1]);
            reqs[3] = __expf(S_rmem[mma_row_id][mma_col_id][3] - max_prev[mma_row_id][1]);

            cur_rowsumexp[0] = reqs[0] + reqs[1];
            cur_rowsumexp[1] = reqs[2] + reqs[2];

            P_rmem[mma_row_id][mma_col_id][0] = __float2bfloat16(reqs[0]);
            P_rmem[mma_row_id][mma_col_id][1] = __float2bfloat16(reqs[1]);
            P_rmem[mma_row_id][mma_col_id][2] = __float2bfloat16(reqs[2]);
            P_rmem[mma_row_id][mma_col_id][3] = __float2bfloat16(reqs[3]);
        }
    }

    // Load V tile into shared memory.
    global_to_shared_copy_by_tile_in_bf16<BLOCK_N, HEAD_DIM, BLOCK_DIM>(v_tile, v_addr_smem, kv_seq_len, kv_start_row, v_stride_n);
    // load V on registers
    for (int mma_row_id = 0; mma_row_id < BLOCK_N / WMMA_N; ++mma_row_id) {
        for (int mma_col_id = 0; mma_col_id < HEAD_DIM / WMMA_K; ++mma_col_id) {
            // m16n8k16 support only row, col format need transpose
            int row = mma_row_id * WMMA_N + (lane_id / 8);
            int col = mma_col_id * WMMA_K + (lane_id / 8) * 8;

            int v_addr_smem_src = v_addr_smem + (row * HEAD_DIM + col) * sizeof(__nv_bfloat16);
            ldmatrix_x2_transpose(V_rmem[mma_row_id][mma_col_id], v_addr_smem_src);
        }
    }

    // ---- Tensor Core GEMM: O += P @ V ----
    // A = P : BLOCK_M x BLOCK_N, row-major
    // B = V : BLOCK_N x HEAD_DIM, row-major
    // C = O : BLOCK_M x HEAD_DIM, row-major
    for (int mma_row_id = 0; mma_row_id < BLOCK_M / WMMA_M; ++mma_row_id) {
        for (int mma_col_id = 0; mma_col_id < HEAD_DIM / WMMA_K; ++mma_col_id) {
            mma_m16n8k16(P_rmem[mma_row_id][mma_col_id],
                         V_rmem[mma_row_id][mma_col_id],
                         O_rmem[mma_row_id][mma_col_id]);
        }
    }
  }

  // Finalize and write back O.
  for (int mma_row_id = 0; mma_row_id < BLOCK_M / WMMA_M; ++mma_row_id) {
      for (int mma_col_id = 0; mma_col_id < HEAD_DIM / WMMA_K; ++mma_col_id) {
          float* reqs = O_rmem[mma_row_id][mma_col_id];
          reqs[0] /= row_sumexp[mma_row_id][0];
          reqs[1] /= row_sumexp[mma_row_id][0];
          reqs[2] /= row_sumexp[mma_row_id][1];
          reqs[3] /= row_sumexp[mma_row_id][1];

          // write back
          const int group_id = lane_id >> 2;
          const int thread_in_group = lane_id & 3;

          const int local_row0 = mma_row_id * WMMA_M + group_id;
          const int local_row1 = local_row0 + 8;
          const int col = mma_col_id * WMMA_N + thread_in_group * 2;

          if (global_start_row + local_row0 < q_seq_len) {
              o_tile[local_row0 * o_stride_n + col + 0] = __float2bfloat16(reqs[0]);
              o_tile[local_row0 * o_stride_n + col + 1] = __float2bfloat16(reqs[1]);
          }
          if (global_start_row + local_row1 < q_seq_len) {
              o_tile[local_row1 * o_stride_n + col + 0] = __float2bfloat16(reqs[2]);
              o_tile[local_row1 * o_stride_n + col + 1] = __float2bfloat16(reqs[3]);
          }
      }
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
  // auto l_tensor = torch::zeros(
  //     {batch_size, num_heads, q_seq_len},
  //     query.options().dtype(torch::kFloat32));

  // TORCH_CHECK(l_tensor.is_cuda());
  // TORCH_CHECK(l_tensor.is_contiguous());
  // TORCH_CHECK(l_tensor.dtype() == torch::kFloat32);

  // auto m_tensor = torch::full(
  //     {batch_size, num_heads, q_seq_len},
  //     -std::numeric_limits<float>::infinity(),
  //     query.options().dtype(torch::kFloat32));

  // TORCH_CHECK(m_tensor.is_cuda());
  // TORCH_CHECK(m_tensor.is_contiguous());
  // TORCH_CHECK(m_tensor.dtype() == torch::kFloat32);

  constexpr int kBlockM = 32;
  constexpr int kBlockN = 64;
  constexpr int kHeadDim = 64;
  TORCH_CHECK(query.size(3) == kHeadDim, "expected head_dim == 64");
  const int num_m_block = (q_seq_len + kBlockM - 1) / kBlockM;

  dim3 grid(num_m_block, num_heads, batch_size);
  dim3 block(BLOCK_DIM);

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
  // C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output_tensor;
}
