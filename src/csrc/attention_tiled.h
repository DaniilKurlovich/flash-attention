#pragma once

#include <c10/util/Optional.h>
#include <torch/torch.h>

torch::Tensor attention_tiled_online_softmax(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal = false,
    c10::optional<double> scale = c10::nullopt,
    int64_t tile_size = 64);
