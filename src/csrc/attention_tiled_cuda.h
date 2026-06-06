#pragma once

#include <torch/torch.h>

torch::Tensor attention_tiled_online_softmax_cuda(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal,
    c10::optional<double> scale,
    int64_t tile_size);
