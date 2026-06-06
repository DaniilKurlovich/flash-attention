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

  std::ostringstream message;
  message << actual_expr << " did not match " << expected_expr
          << " (atol=" << atol << ", rtol=" << rtol
          << ", max_abs=" << max_abs << ", max_rel=" << max_rel << ")";
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
