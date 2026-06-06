#include <cmath>

#include <torch/cuda.h>
#include <torch/torch.h>

#include "../attention_tiled_cuda.h"
#include "test_harness.h"

namespace {

torch::Tensor attention_reference_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal,
    c10::optional<double> scale) {
  const double softmax_scale = scale.has_value()
      ? *scale
      : 1.0 / std::sqrt(static_cast<double>(query.size(3)));

  auto scores = torch::matmul(
                    query.to(torch::kFloat32),
                    key.to(torch::kFloat32).transpose(-2, -1)) *
      softmax_scale;

  if (causal) {
    const auto q_positions =
        torch::arange(query.size(2), query.options().dtype(torch::kLong));
    const auto k_positions =
        torch::arange(key.size(2), key.options().dtype(torch::kLong));
    const auto mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(-1);
    scores = scores.masked_fill(
        mask.logical_not().unsqueeze(0).unsqueeze(0),
        -std::numeric_limits<float>::infinity());
  }

  return torch::matmul(torch::softmax(scores, -1), value.to(torch::kFloat32)).to(torch::kFloat16);
}

torch::Tensor make_cuda_tensor(int batch_size = 8, int num_heads = 8, int seq_len = 128, int head_dim = 64) {
  return torch::randn({batch_size, num_heads, seq_len, head_dim},
                      torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat16));
}

}  // namespace

FLASH_ATTENTION_TEST(cuda_matches_reference_non_causal) {
  if (!torch::cuda::is_available()) {
    FLASH_ATTENTION_SKIP("CUDA is not available at runtime");
  }

  const auto query = make_cuda_tensor();
  const auto key = make_cuda_tensor();
  const auto value = make_cuda_tensor();

  const auto actual = attention_tiled_online_softmax_cuda(
      query,
      key,
      value,
      false,
      c10::nullopt,
      8);
  const auto expected =
      attention_reference_cuda(query, key, value, false, c10::nullopt);

  FLASH_ATTENTION_ASSERT(actual.is_cuda());
  FLASH_ATTENTION_ASSERT(actual.scalar_type() == torch::kFloat16);
  FLASH_ATTENTION_ASSERT_TENSOR_CLOSE(
      actual.cpu().to(torch::kFloat32),
      expected.cpu().to(torch::kFloat32),
      2e-3,
      1e-3);
}

// FLASH_ATTENTION_TEST(cuda_matches_reference_causal_with_custom_scale) {
//   if (!torch::cuda::is_available()) {
//     FLASH_ATTENTION_SKIP("CUDA is not available at runtime");
//   }

//   const auto query = make_cuda_tensor();
//   const auto key = make_cuda_tensor();
//   const auto value = make_cuda_tensor();

//   const auto actual =
//       attention_tiled_online_softmax_cuda(query, key, value, true, 0.125, 8);
//   const auto expected = attention_reference_cuda(query, key, value, true, 0.125);

//   FLASH_ATTENTION_ASSERT_TENSOR_CLOSE(actual.cpu(), expected.cpu(), 5e-2, 5e-2);
// }
