#include "test_harness.h"

#include <iostream>
#include <sstream>

namespace flash_attention::tests {

std::vector<TestCase>& registry() {
  static std::vector<TestCase> test_cases;
  return test_cases;
}

TestRegistrar::TestRegistrar(const char* name, TestFunction function) {
  registry().push_back({name, function});
}

std::string tensor_mismatch_message(
    const char* actual_expr,
    const char* expected_expr,
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    double atol,
    double rtol) {
  const auto actual_fp32 = actual.to(torch::kFloat32);
  const auto expected_fp32 = expected.to(torch::kFloat32);
  const auto diff = (actual_fp32 - expected_fp32).abs();
  const double max_abs = diff.max().item<double>();
  const double max_rel = (diff / expected_fp32.abs().clamp_min(1e-12)).max().item<double>();

  const auto flat_idx = diff.argmax().item<int64_t>();
  const auto idx = diff.reshape(-1);  // dummy, will use indexing
  std::vector<int64_t> coords(actual.dim());
  auto temp = diff;
  auto rem = flat_idx;
  for (int64_t i = actual.dim() - 1; i >= 0; --i) {
    coords[i] = rem % actual.size(i);
    rem /= actual.size(i);
  }
  std::ostringstream idx_str;
  idx_str << "[";
  for (size_t i = 0; i < coords.size(); ++i) {
    idx_str << coords[i];
    if (i + 1 < coords.size()) idx_str << ", ";
  }
  idx_str << "]";
  const float actual_val = actual_fp32.reshape(-1)[flat_idx].item<float>();
  const float expected_val = expected_fp32.reshape(-1)[flat_idx].item<float>();

  std::ostringstream message;
  message << actual_expr << " did not match " << expected_expr
          << " (atol=" << atol << ", rtol=" << rtol
          << ", max_abs=" << max_abs << ", max_rel=" << max_rel << ")"
          << " max error at " << idx_str.str()
          << " actual=" << actual_val << " expected=" << expected_val;
  return message.str();
}

[[noreturn]] void fail(const std::string& message) {
  throw TestFailure(message);
}

[[noreturn]] void skip(const std::string& message) {
  throw TestSkipped(message);
}

int run_all_tests() {
  int passed = 0;
  int skipped = 0;
  int failed = 0;

  for (const auto& test_case : registry()) {
    try {
      test_case.function();
      ++passed;
      std::cout << "[PASS] " << test_case.name << '\n';
    } catch (const TestSkipped& error) {
      ++skipped;
      std::cout << "[SKIP] " << test_case.name << ": " << error.what() << '\n';
    } catch (const std::exception& error) {
      ++failed;
      std::cerr << "[FAIL] " << test_case.name << ": " << error.what() << '\n';
    }
  }

  std::cout << "Summary: " << passed << " passed, " << skipped << " skipped, "
            << failed << " failed" << '\n';
  return failed == 0 ? 0 : 1;
}

}  // namespace flash_attention::tests
