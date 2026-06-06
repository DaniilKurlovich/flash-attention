#include <pybind11/pybind11.h>

#include <torch/extension.h>

#include "attention_tiled.h"

namespace py = pybind11;

namespace {

c10::optional<double> parse_scale(const py::object& scale_obj) {
  if (scale_obj.is_none()) {
    return c10::nullopt;
  }
  return scale_obj.cast<double>();
}

torch::Tensor attention_tiled_online_softmax_python(
    const torch::Tensor& query,
    const torch::Tensor& key,
    const torch::Tensor& value,
    bool causal,
    py::object scale_obj,
    int64_t tile_size) {
  return attention_tiled_online_softmax(
      query,
      key,
      value,
      causal,
      parse_scale(scale_obj),
      tile_size);
}

}  // namespace

PYBIND11_MODULE(flash_attention_cpp, module) {
  module.doc() = "C++ wrapper for tiled Flash Attention reference kernels";
  module.def(
      "attention_tiled_online_softmax",
      &attention_tiled_online_softmax_python,
      py::arg("query"),
      py::arg("key"),
      py::arg("value"),
      py::kw_only(),
      py::arg("causal") = false,
      py::arg("scale") = py::none(),
      py::arg("tile_size") = 64);
}
