#include <cmath>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "../attention_tiled.h"
#include "test_harness.h"

namespace {

torch::Tensor attention_reference(
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
    const auto q_positions = torch::arange(query.size(2), torch::kLong);
    const auto k_positions = torch::arange(key.size(2), torch::kLong);
    const auto mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(-1);
    scores = scores.masked_fill(
        mask.logical_not().unsqueeze(0).unsqueeze(0),
        -std::numeric_limits<float>::infinity());
  }

  return torch::matmul(torch::softmax(scores, -1), value.to(torch::kFloat32))
      .to(value.scalar_type());
}

torch::Tensor make_tensor(const std::vector<int64_t>& shape, double start) {
  const auto numel = std::accumulate(shape.begin(), shape.end(), int64_t{1}, std::multiplies<int64_t>());
  return torch::arange(
             start,
             start + static_cast<double>(numel),
             torch::TensorOptions().dtype(torch::kFloat32))
      .reshape(shape) /
      10.0;
}

}  // namespace

FLASH_ATTENTION_TEST(cpu_matches_reference_non_causal) {
  const auto query = make_tensor({1, 2, 5, 4}, 0.0);
  const auto key = make_tensor({1, 2, 5, 4}, 4.0);
  const auto value = make_tensor({1, 2, 5, 3}, 8.0);

  const auto actual =
      attention_tiled_online_softmax(query, key, value, false, c10::nullopt, 2);
  const auto expected =
      attention_reference(query, key, value, false, c10::nullopt);

  FLASH_ATTENTION_ASSERT_TENSOR_CLOSE(actual, expected, 1e-5, 1e-5);
}

FLASH_ATTENTION_TEST(cpu_matches_reference_causal_with_custom_scale) {
  const auto query = make_tensor({1, 1, 4, 4}, -2.0);
  const auto key = make_tensor({1, 1, 4, 4}, 3.0);
  const auto value = make_tensor({1, 1, 4, 2}, 6.0);

  const auto actual =
      attention_tiled_online_softmax(query, key, value, true, 0.25, 3);
  const auto expected = attention_reference(query, key, value, true, 0.25);

  FLASH_ATTENTION_ASSERT_TENSOR_CLOSE(actual, expected, 1e-5, 1e-5);
}

FLASH_ATTENTION_TEST(rejects_non_positive_tile_size) {
  const auto query = torch::zeros({1, 1, 2, 2}, torch::kFloat32);
  bool threw = false;

  try {
    (void)attention_tiled_online_softmax(
        query,
        query,
        query,
        false,
        c10::nullopt,
        0);
  } catch (const std::invalid_argument& error) {
    threw = std::string(error.what()).find("tile_size") != std::string::npos;
  }

  FLASH_ATTENTION_ASSERT(threw);
}
