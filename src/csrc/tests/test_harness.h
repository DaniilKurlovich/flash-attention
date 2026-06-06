#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <torch/torch.h>

namespace flash_attention::tests {

using TestFunction = void (*)();

struct TestCase {
  const char* name;
  TestFunction function;
};

class TestFailure : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class TestSkipped : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class TestRegistrar {
 public:
  TestRegistrar(const char* name, TestFunction function);
};

std::vector<TestCase>& registry();
std::string tensor_mismatch_message(
    const char* actual_expr,
    const char* expected_expr,
    const torch::Tensor& actual,
    const torch::Tensor& expected,
    double atol,
    double rtol);
[[noreturn]] void fail(const std::string& message);
[[noreturn]] void skip(const std::string& message);
int run_all_tests();

}  // namespace flash_attention::tests

#define FLASH_ATTENTION_CONCAT_INNER(x, y) x##y
#define FLASH_ATTENTION_CONCAT(x, y) FLASH_ATTENTION_CONCAT_INNER(x, y)

#define FLASH_ATTENTION_TEST(name)                                            \
  static void name();                                                         \
  static ::flash_attention::tests::TestRegistrar FLASH_ATTENTION_CONCAT(      \
      name,                                                                  \
      _registrar)(#name, &name);                                             \
  static void name()

#define FLASH_ATTENTION_ASSERT(condition)                                     \
  do {                                                                        \
    if (!(condition)) {                                                       \
      ::flash_attention::tests::fail("assertion failed: " #condition);        \
    }                                                                         \
  } while (false)

#define FLASH_ATTENTION_SKIP(message) ::flash_attention::tests::skip(message)

#define FLASH_ATTENTION_ASSERT_TENSOR_CLOSE(actual, expected, atol, rtol)     \
  do {                                                                        \
    const auto flash_attention_actual_tensor = (actual);                      \
    const auto flash_attention_expected_tensor = (expected);                  \
    const auto flash_attention_actual_fp32 =                                  \
        flash_attention_actual_tensor.to(torch::kFloat32);                    \
    const auto flash_attention_expected_fp32 =                                \
        flash_attention_expected_tensor.to(torch::kFloat32);                  \
    if (!flash_attention_actual_fp32.allclose(                                \
            flash_attention_expected_fp32,                                    \
            rtol,                                                             \
            atol)) {                                                          \
      ::flash_attention::tests::fail(                                         \
          ::flash_attention::tests::tensor_mismatch_message(                  \
              #actual,                                                        \
              #expected,                                                      \
              flash_attention_actual_tensor,                                  \
              flash_attention_expected_tensor,                                \
              atol,                                                           \
              rtol));                                                         \
    }                                                                         \
  } while (false)
