#include <cmath>
#include <iostream>

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
        torch::arange(key.size(2), query.options().dtype(torch::kLong));
    const auto mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(-1);
    scores = scores.masked_fill(
        mask.logical_not().unsqueeze(0).unsqueeze(0),
        -std::numeric_limits<float>::infinity());
  }

  return torch::matmul(torch::softmax(scores, -1), value.to(torch::kFloat32)).to(torch::kBFloat16);
}

}  // namespace

FLASH_ATTENTION_TEST(cuda_matches_reference_debug) {
  if (!torch::cuda::is_available()) {
    FLASH_ATTENTION_SKIP("CUDA is not available at runtime");
  }

  torch::manual_seed(0);
  const auto query = torch::randn({1, 1, 64, 64},
                                  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
  const auto key = torch::randn({1, 1, 64, 64},
                                torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));
  const auto value = torch::randn({1, 1, 64, 64},
                                  torch::TensorOptions().device(torch::kCUDA).dtype(torch::kBFloat16));

  const auto actual = attention_tiled_online_softmax_cuda(
      query, key, value, false, c10::nullopt, 8);
  const auto expected = attention_reference_cuda(query, key, value, false, c10::nullopt);

  const auto diff = (actual.cpu().to(torch::kFloat32) - expected.cpu().to(torch::kFloat32)).abs();
  std::cout << "max_abs=" << diff.max().item<float>()
            << " mean_abs=" << diff.mean().item<float>() << "\n";

  // Compare a pure torch reference on the same inputs (bf16 all the way through)
  const auto expected_bf16 = torch::matmul(
      torch::softmax(
          torch::matmul(query, key.transpose(-2, -1)) / std::sqrt(64.0), -1),
      value);
  const auto diff2 = (actual.cpu().to(torch::kFloat32) - expected_bf16.cpu().to(torch::kFloat32)).abs();
  std::cout << "vs bf16 matmul max_abs=" << diff2.max().item<float>()
            << " mean_abs=" << diff2.mean().item<float>() << "\n";

  // Print first row of output
  std::cout << "actual[0,0,0,:8]:  ";
  for (int i = 0; i < 8; ++i) {
    std::cout << actual[0][0][0][i].item<float>() << " ";
  }
  std::cout << "\nexpected[0,0,0,:8]: ";
  for (int i = 0; i < 8; ++i) {
    std::cout << expected[0][0][0][i].item<float>() << " ";
  }
  std::cout << "\n";
}
