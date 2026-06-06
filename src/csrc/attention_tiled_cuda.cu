#include <cuda_runtime.h>

#include <cmath>
#include <limits>

#include <c10/cuda/CUDAException.h>
#include <torch/torch.h>

#include "attention_tiled_cuda.h"

namespace {


template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void attention_tiled_online_softmax_kernel_stub_v2(
  const half* __restrict__ query_ptr,
  const half* __restrict__ key_ptr,
  const half* __restrict__ value_ptr,
  half* __restrict__ output_ptr,
  // float* __restrict__ l_ptr,
  // float* __restrict__ m_ptr,

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
  __shared__ half q_smem[BLOCK_M][HEAD_DIM];
  __shared__ half k_smem[BLOCK_N][HEAD_DIM];
  __shared__ half v_smem[BLOCK_N][HEAD_DIM];

  __shared__ float s_smem[BLOCK_M][BLOCK_N];
  __shared__ float o_smem[BLOCK_M][HEAD_DIM];

  __shared__ float l_smem[BLOCK_M];
  __shared__ float  m_smem[BLOCK_M];

  int batch_id = blockIdx.z;
  int head_id = blockIdx.y;
  int tile_id = blockIdx.x;

  const half* q_tile = query_ptr +
               batch_id * q_stride_b +
               head_id * q_stride_h +
               tile_id * BLOCK_M * q_stride_n;

  const half* k_tile = key_ptr +
               batch_id * k_stride_b +
               head_id * k_stride_h;
  
  const half* v_tile = value_ptr +
               batch_id * v_stride_b +
               head_id * v_stride_h;

  half* o_tile = output_ptr +
               batch_id * o_stride_b +
               head_id * o_stride_h +
               tile_id * BLOCK_M * o_stride_n;
  
  // load q once
  for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
    int row = i / HEAD_DIM;
    int col = i % HEAD_DIM;
    int global_index_row = tile_id * BLOCK_M + row;
    q_smem[row][col] = (global_index_row < q_seq_len) ? q_tile[row * q_stride_n + col]: __float2half(0.0f);
  }

  for (int i = threadIdx.x; i < BLOCK_M * HEAD_DIM; i += blockDim.x) {
    int row = i / HEAD_DIM;
    int col = i % HEAD_DIM;
    o_smem[row][col] = 0.0f;
  }

  // prepare l_smem, m_smem
  for (int row = threadIdx.x; row < BLOCK_M; row += blockDim.x) {
    l_smem[row] = 0;
    m_smem[row] = -INFINITY;
  }
  __syncthreads();

  const int iter_loops = (kv_seq_len + BLOCK_N - 1) / BLOCK_N;
  for (int kv_tile_id = 0; kv_tile_id < iter_loops; ++kv_tile_id) {
    
    int kv_start_row = BLOCK_N * kv_tile_id;
    // load k tile in SRAM
    for (int i = threadIdx.x; i < BLOCK_N * HEAD_DIM; i += blockDim.x) {
      int row = i / HEAD_DIM;
      int col = i % HEAD_DIM;
      
      int start_tile = kv_start_row + row;
      k_smem[row][col] = (start_tile < kv_seq_len) ? k_tile[start_tile * k_stride_n + col]: __float2half(0.0f);
    }
    __syncthreads();

    // compute S_ij = Q_i * K_j^T
    for (int idx = threadIdx.x; idx < BLOCK_M * BLOCK_N; idx += blockDim.x) {
      int row = idx / BLOCK_N;
      int col = idx % BLOCK_N;

      float s_ij = 0.0f;
      for (int d = 0; d < HEAD_DIM; d++) {
        s_ij = fmaf(__half2float(q_smem[row][d]), __half2float(k_smem[col][d]), s_ij);
      }
      s_ij *= scale;

      int global_q_row = tile_id * BLOCK_M + row;
      int global_kv_row = kv_start_row + col;
      if (causal && global_kv_row > global_q_row) s_ij = -INFINITY;
      if (global_q_row >= q_seq_len || global_kv_row >= kv_seq_len) s_ij = -INFINITY;
      s_smem[row][col] = s_ij;
    }
    __syncthreads();

    // online softmax
    for (int row = threadIdx.x; row < BLOCK_M; row += blockDim.x) {
      // compute m_tile
      float m_tile = -INFINITY;
      for (int col = 0; col < BLOCK_N; ++col) {
        m_tile = fmaxf(m_tile, s_smem[row][col]);
      }

      float max_prev = m_smem[row];
      float max_new = fmaxf(max_prev, m_tile);

      // rescale O
      float scale_old = expf(max_prev - max_new);
      for (int d = 0; d < HEAD_DIM; ++d) {
        o_smem[row][d] *= scale_old;
      }
      l_smem[row] *= scale_old;
      
      float l_tile = 0.0f;
      // compute P~_ij
      for (int col = 0; col < BLOCK_N; ++col) {
        float p_ij = expf(s_smem[row][col] - max_new);
        l_tile += (max_new == -INFINITY) ? 0.0f : p_ij;
        s_smem[row][col] = (max_new == -INFINITY) ? 0.0f : p_ij;
      }

      l_smem[row] += l_tile;
      m_smem[row] = fmaxf(m_smem[row], max_new);
    }
    __syncthreads();

    // load V_j into SRAM
    for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
      int row = idx / HEAD_DIM;
      int col = idx % HEAD_DIM;

      int start_tile = kv_start_row + row;
      v_smem[row][col] = (start_tile < kv_seq_len) 
                          ? v_tile[start_tile * v_stride_n + col]
                          : __float2half(0.0f);
    }
    __syncthreads();

    // last part: O += P~_ij * V_j
    for (int idx = threadIdx.x; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
      int row = idx / HEAD_DIM;
      int d = idx % HEAD_DIM;

      float acc = 0.0f;
      for (int col = 0; col < BLOCK_N; ++col) {
        acc = fmaf(s_smem[row][col], v_smem[col][d], acc);
      }
      o_smem[row][d] += acc;
    }
    
    __syncthreads();
  }

  // finilize and write back O, l, m
  for (int idx = threadIdx.x; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
    int row = idx / HEAD_DIM;
    int col = idx % HEAD_DIM;
    int global_q_row = tile_id * BLOCK_M + row;
    if (global_q_row >= q_seq_len) continue;
    float o = (l_smem[row] > 0.0f) ? (o_smem[row][col] / l_smem[row]) : 0.0f;
    o_tile[row * o_stride_n + col] = __float2half(o);
    o_smem[row][col] /= l_smem[row];
  }

}

template<int BLOCK_M, int BLOCK_N, int HEAD_DIM>
__global__ void attention_tiled_online_softmax_kernel_stub_v1(
  const half* __restrict__ query_ptr,   // [B, num_heads, seq_len, HEAD_DIM]
  const half* __restrict__ key_ptr,
  const half* __restrict__ value_ptr,
  half* __restrict__ output_ptr,
  // float* __restrict__ l_ptr,
  // float* __restrict__ m_ptr,

  int batch_size,
  int num_heads,
  int seq_len,
  
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
  // l2 init
  __shared__ half q_smem[BLOCK_M][HEAD_DIM];
  __shared__ half k_smem[BLOCK_N][HEAD_DIM];
  __shared__ half v_smem[BLOCK_N][HEAD_DIM];
  __shared__ float s_smem[BLOCK_M][BLOCK_N];

  __shared__ float m_smem[BLOCK_M];
  __shared__ float m_tilda_smem[BLOCK_M];
  __shared__ float m_new_smem[BLOCK_M];

  __shared__ float l_smem[BLOCK_M];
  __shared__ float l_tilda_smem[BLOCK_M];
  __shared__ float P_ij_smem[BLOCK_M][BLOCK_N];
  
  __shared__ float o_smem[BLOCK_M][HEAD_DIM];
  //
  int batchId = blockIdx.z;
  int headId = blockIdx.y;
  int qTileId = blockIdx.x;

  const half* q_block = query_ptr +
                        batchId * q_stride_b +
                        headId * q_stride_h +
                        qTileId * BLOCK_M * q_stride_n;
  
  const half* k_block = key_ptr + 
                        batchId * k_stride_b +
                        headId * k_stride_h;

  const half* v_block = value_ptr +
                        batchId * v_stride_b +
                        headId * v_stride_h;
  
  const auto packed_global_offset = (batchId * num_heads + headId) * seq_len + qTileId * BLOCK_M;
  // const float* l_block = l_ptr + packed_global_offset;
  // const float* m_block = m_ptr + packed_global_offset;

  // 0-step: prepare caches
  for (int idx = threadIdx.x; idx < BLOCK_M; idx += blockDim.x) {
    int row = idx;
    int global_q_row = qTileId * BLOCK_M + row;
    m_smem[idx] = -INFINITY;
    l_smem[idx] = 0.0f;
    // m_smem[idx] = (global_q_row < seq_len)
    //               ? m_block[row]
    //               : -INFINITY;
    // l_smem[idx] = (global_q_row < seq_len)
    //               ? l_block[row]
    //               : 0.0f;
  }

  for (int idx = threadIdx.x; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
    int row = idx / HEAD_DIM;
    int col = idx % HEAD_DIM;
    o_smem[row][col] = 0.0f;
  }
  __syncthreads();

  // 1-step: for simplicity load only q_tile in cache
  for (int idx = threadIdx.x; idx < BLOCK_M * HEAD_DIM; idx += blockDim.x) {
    int row = idx / HEAD_DIM;
    int col = idx % HEAD_DIM;
    int global_q_row = qTileId * BLOCK_M + row;

    q_smem[row][col] = (global_q_row < seq_len) 
                        ? q_block[row * q_stride_n + col]
                        : __float2half(0.0f);
  }
  __syncthreads();

  const int outer_loop_iters = (seq_len + BLOCK_N - 1) / BLOCK_N;
  for (int kv_tile_id = 0; kv_tile_id < outer_loop_iters; kv_tile_id++) {
    int kv_start = kv_tile_id * BLOCK_N;

    // 2-step: store tile K, V in sram
    for (int idx = threadIdx.x; idx < BLOCK_N * HEAD_DIM; idx += blockDim.x) {
      int row = idx / HEAD_DIM;
      int col = idx % HEAD_DIM;
      int global_kv_row = kv_start + row;

      k_smem[row][col] = (global_kv_row < seq_len)
                          ? k_block[global_kv_row * k_stride_n + col]
                          : __float2half(0.0f);

      v_smem[row][col] = (global_kv_row < seq_len)
                          ? v_block[global_kv_row * v_stride_n + col]
                          : __float2half(0.0f);
    }
    __syncthreads();

    // 3-step: compute score
    for (int linear = threadIdx.x; linear < BLOCK_M * BLOCK_N; linear += blockDim.x) {
      // TODO: make accurate 
      int i = linear / BLOCK_N;
      int j = linear % BLOCK_N;

      int global_q_row = qTileId * BLOCK_M + i;
      int global_k_row = kv_start + j;

      float s_ij = 0.0f;
      for (int d = 0; d < HEAD_DIM; ++d) {
        s_ij = fmaf(__half2float(q_smem[i][d]), __half2float(k_smem[j][d]), s_ij);
      }
      s_ij *= scale;

      if (global_q_row >= seq_len || global_k_row >= seq_len) s_ij = -INFINITY;
      if (causal && global_k_row > global_q_row) s_ij = -INFINITY;

      s_smem[i][j] = s_ij;
    }
    __syncthreads();

    // 4-step: compute ~m = rowmax(S_ij) for this KV tile
    for (int row = threadIdx.x; row < BLOCK_M; row += blockDim.x) {
      float row_max = -INFINITY;
      for (int col = 0; col < BLOCK_N; ++col) {
        row_max = fmaxf(row_max, s_smem[row][col]);
      }
      m_tilda_smem[row] = row_max;
    }
    __syncthreads();

    // 5-step: combine running m with tile-local ~m into m_new,
    // and compute alpha = exp(m_prev - m_new) for rescaling O_prev and l_prev.
    // Using m_new directly in the exp below avoids needing a separate beta.
    for (int row = threadIdx.x; row < BLOCK_M; row += blockDim.x) {
      float m_prev  = m_smem[row];
      float m_tilda = m_tilda_smem[row];
      float m_new   = fmaxf(m_prev, m_tilda);
      // If m_prev == -inf (first tile), alpha = 0 so old state contributes nothing.
      float alpha   = (m_prev == -INFINITY) ? 0.0f : __expf(m_prev - m_new);
      m_new_smem[row] = m_new;
      // stash alpha by overwriting m_tilda_smem — we no longer need m_tilda past this point.
      m_tilda_smem[row] = alpha;
    }
    __syncthreads();

    // 6-step: compute P_ij = exp(S_ij - m_new) (combined, numerically stable)
    for (int linear = threadIdx.x; linear < BLOCK_M * BLOCK_N; linear += blockDim.x) {
      int i = linear / BLOCK_N;
      int j = linear % BLOCK_N;
      float m_new = m_new_smem[i];
      P_ij_smem[i][j] = (m_new == -INFINITY) ? 0.0f
                                             : __expf(s_smem[i][j] - m_new);
    }
    __syncthreads();

    // 7-step: l_new = alpha * l_prev + rowsum(P_ij), then commit m,l running state.
    for (int row = threadIdx.x; row < BLOCK_M; row += blockDim.x) {
      float row_sum = 0.0f;
      for (int col = 0; col < BLOCK_N; ++col) {
        row_sum += P_ij_smem[row][col];
      }
      float alpha = m_tilda_smem[row];   // stashed above
      l_smem[row] = alpha * l_smem[row] + row_sum;
      m_smem[row] = m_new_smem[row];
    }
    __syncthreads();

    // 8-step: O = alpha * O + P @ V   (rescale old output, then accumulate P·V)
    // Each thread handles one (i, d) cell of the BLOCK_M x HEAD_DIM output tile.
    for (int linear = threadIdx.x; linear < BLOCK_M * HEAD_DIM; linear += blockDim.x) {
      int i = linear / HEAD_DIM;
      int d = linear % HEAD_DIM;
      float alpha = m_tilda_smem[i];
      float acc   = alpha * o_smem[i][d];
      for (int j = 0; j < BLOCK_N; ++j) {
        acc = fmaf(P_ij_smem[i][j], __half2float(v_smem[j][d]), acc);
      }
      o_smem[i][d] = acc;
    }
    __syncthreads();
  }

  // 9-step: finalize. Normalize O by running l, write O to global.
  // Also stash LSE = m + log(l) and l for the backward pass.
  for (int linear = threadIdx.x; linear < BLOCK_M * HEAD_DIM; linear += blockDim.x) {
    int i = linear / HEAD_DIM;
    int d = linear % HEAD_DIM;
    int global_q_row = qTileId * BLOCK_M + i;
    if (global_q_row >= seq_len) continue;

    float l = l_smem[i];
    float o = (l > 0.0f) ? (o_smem[i][d] / l) : 0.0f;
    output_ptr[batchId * o_stride_b
             + headId  * o_stride_h
             + global_q_row * o_stride_n
             + d] = __float2half(o);
  }

  for (int row = threadIdx.x; row < BLOCK_M; row += blockDim.x) {
    int global_q_row = qTileId * BLOCK_M + row;
    if (global_q_row >= seq_len) continue;
    float m = m_smem[row];
    float l = l_smem[row];
    int idx = (batchId * num_heads + headId) * seq_len + global_q_row;
    // l_ptr[idx] = l;
    // m_ptr[idx] = (l > 0.0f) ? (m + logf(l)) : -INFINITY;  // LSE
  }
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
  TORCH_CHECK(
      query.scalar_type() == torch::kFloat16,
      "query must be a Float16 dtype tensor");

  TORCH_CHECK(key.is_cuda(), "key must be a CUDA tensor");
  TORCH_CHECK(
      key.scalar_type() == torch::kFloat16,
      "key must be a Float16 dtype tensor");

  TORCH_CHECK(value.is_cuda(), "value must be a CUDA tensor");
  TORCH_CHECK(
      value.scalar_type() == torch::kFloat16,
      "value must be a Float16 dtype tensor");
  
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
    query.options().dtype(torch::kFloat16)
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

  auto query_ptr = reinterpret_cast<const half*>(query.data_ptr<c10::Half>());
  auto key_ptr   = reinterpret_cast<const half*>(key.data_ptr<c10::Half>());
  auto value_ptr = reinterpret_cast<const half*>(value.data_ptr<c10::Half>());
  auto output_ptr = reinterpret_cast<half*>(output_tensor.data_ptr<c10::Half>());
  auto l_ptr = l_tensor.data_ptr<float>();
  auto m_ptr = m_tensor.data_ptr<float>();
  
  constexpr int kBlockM = 32;
  constexpr int kBlockN = 64;
  constexpr int kHeadDim = 64;
  TORCH_CHECK(query.size(3) == kHeadDim, "expected head_dim == 64");
  const int num_m_block = (q_seq_len + kBlockM - 1) / kBlockM;

  dim3 grid(num_m_block, num_heads, batch_size);
  dim3 block(256);
  attention_tiled_online_softmax_kernel_stub_v2<kBlockM, kBlockN, kHeadDim><<<grid, block>>>(
        query_ptr,
        key_ptr,
        value_ptr,
        output_ptr,
        // l_ptr,
        // m_ptr,
        batch_size,
        num_heads,
        q_seq_len,
        k_seq_len,
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
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  return output_tensor;
}
