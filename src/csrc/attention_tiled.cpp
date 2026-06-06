#include <cmath>
#include <limits>
#include <stdexcept>

#include "attention_tiled.h"
#include "attention_tiled_cuda.h"
using torch::indexing::Slice;

#if !FLASH_ATTENTION_ENABLE_CUDA
torch::Tensor attention_tiled_online_softmax_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal,
    c10::optional<double> scale,
    int64_t tile_size) {
  (void)query;
  (void)key;
  (void)value;
  (void)causal;
  (void)scale;
  (void)tile_size;
  TORCH_CHECK(
      false,
      "flash_attention_cpp was built without CUDA support. "
      "Reconfigure with -DFLASH_ATTENTION_ENABLE_CUDA=ON to run CUDA tensors.");
}
#endif

namespace {

void validate_inputs(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value) {
  if (query.dim() != 4 || key.dim() != 4 || value.dim() != 4) {
    throw std::invalid_argument(
        "query, key, and value must all be 4D tensors shaped (B, H, T, D)");
  }

  if (query.size(0) != key.size(0) || query.size(1) != key.size(1) ||
      query.size(0) != value.size(0) || query.size(1) != value.size(1)) {
    throw std::invalid_argument(
        "query, key, and value must agree on batch and head dimensions");
  }

  if (key.size(2) != value.size(2)) {
    throw std::invalid_argument("key and value must have the same sequence length");
  }

  if (key.size(3) != query.size(3)) {
    throw std::invalid_argument("query and key must have the same head dimension");
  }

  if (value.size(3) <= 0) {
    throw std::invalid_argument("value head dimension must be positive");
  }
}

}  // namespace

torch::Tensor attention_tiled_online_softmax(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal,
    c10::optional<double> scale,
    int64_t tile_size) {
  validate_inputs(query, key, value);
  if (tile_size <= 0) {
    throw std::invalid_argument("tile_size must be positive");
  }

  if (query.is_cuda() || key.is_cuda() || value.is_cuda()) {
    TORCH_CHECK(
        query.is_cuda() && key.is_cuda() && value.is_cuda(),
        "query, key, and value must all be CUDA tensors when using the CUDA path");
    return attention_tiled_online_softmax_cuda(
        query,
        key,
        value,
        causal,
        scale,
        tile_size);
  }

  const auto q_len = query.size(2);
  const auto k_len = key.size(2);
  const auto value_dim = value.size(3);
  const double softmax_scale = scale.has_value()
      ? *scale
      : 1.0 / std::sqrt(static_cast<double>(query.size(3)));

  const auto q = query.to(torch::kFloat32);
  const auto k = key.to(torch::kFloat32);
  const auto v = value.to(torch::kFloat32);

  auto accum_options = query.options().dtype(torch::kFloat32);
  auto output =
      torch::empty({query.size(0), query.size(1), q_len, value_dim}, accum_options);
  auto row_max = torch::full(
      {query.size(0), query.size(1), q_len},
      -std::numeric_limits<float>::infinity(),
      accum_options);
  auto row_sum = torch::zeros({query.size(0), query.size(1), q_len}, accum_options);
  auto row_acc =
      torch::zeros({query.size(0), query.size(1), q_len, value_dim}, accum_options);

  torch::Tensor q_positions;
  if (causal) {
    q_positions = torch::arange(q_len, query.options().dtype(torch::kLong));
  }

  for (int64_t start = 0; start < k_len; start += tile_size) {
    const auto end = std::min(start + tile_size, k_len);
    const auto k_tile = k.index({Slice(), Slice(), Slice(start, end), Slice()});
    const auto v_tile = v.index({Slice(), Slice(), Slice(start, end), Slice()});

    auto scores = torch::matmul(q, k_tile.transpose(-2, -1)) * softmax_scale;
    if (causal) {
      const auto k_positions =
          torch::arange(start, end, query.options().dtype(torch::kLong));
      const auto tile_mask =
          k_positions.unsqueeze(0) <= q_positions.unsqueeze(-1);
      scores = scores.masked_fill(
          tile_mask.logical_not().unsqueeze(0).unsqueeze(0),
          -std::numeric_limits<float>::infinity());
    }

    const auto tile_row_max = std::get<0>(scores.max(-1));
    const auto new_row_max = torch::maximum(row_max, tile_row_max);

    const auto exp_row_scale = torch::exp(row_max - new_row_max);
    const auto exp_scores = torch::exp(scores - new_row_max.unsqueeze(-1));

    row_sum = row_sum * exp_row_scale + exp_scores.sum(-1);
    row_acc =
        row_acc * exp_row_scale.unsqueeze(-1) + torch::matmul(exp_scores, v_tile);
    row_max = new_row_max;
  }

  output.copy_(row_acc / row_sum.unsqueeze(-1));
  return output.to(value.scalar_type());
}
