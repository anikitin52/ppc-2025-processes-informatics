#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"
#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinARunFuncTestsProcesses : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    // Получаем параметры теста
    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    int test_case = std::get<0>(params);
    
    // Создаем тестовые векторы для разных случаев
    switch (test_case) {
      case 0: // Все положительные - 0 чередований
        input_data_ = {1, 2, 3, 4, 5};
        expected_output_ = 0;
        break;
      case 1: // Чередование +-+- - 4 чередования
        input_data_ = {1, -2, 3, -4, 5};
        expected_output_ = 4;
        break;
      case 2: // Чередование -+-+ - 4 чередования  
        input_data_ = {-1, 2, -3, 4, -5};
        expected_output_ = 4;
        break;
      case 3: // Случай с нулями (0 считается положительным)
        input_data_ = {1, 0, -1, 0, 1};
        expected_output_ = 2;
        break;
      case 4: // Один элемент - 0 чередований
        input_data_ = {5};
        expected_output_ = 0;
        break;
      case 5: // Два элемента с разными знаками
        input_data_ = {-1, 1};
        expected_output_ = 1;
        break;
      case 6: // Два элемента с одинаковыми знаками
        input_data_ = {1, 2};
        expected_output_ = 0;
        break;
      case 7: // Длинная последовательность
        input_data_ = {1, -1, 1, -1, 1, -1, 1, -1};
        expected_output_ = 7;
        break;
      case 8: // Смешанная последовательность
        input_data_ = {5, -3, 2, -8, 7, -1, 4, -6};
        expected_output_ = 7;
        break;
      default:
        input_data_ = {1, -1};
        expected_output_ = 1;
        break;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (expected_output_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
  OutType expected_output_;
};

namespace {

TEST_P(NikitinARunFuncTestsProcesses, SignAlternationTest) {
  ExecuteTest(GetParam());
}

// Тестовые случаи: (номер_теста, описание)
const std::array<TestType, 9> kTestParam = {
  std::make_tuple(0, "all_positive"),
  std::make_tuple(1, "alternating_positive_negative"), 
  std::make_tuple(2, "alternating_negative_positive"),
  std::make_tuple(3, "with_zeros"),
  std::make_tuple(4, "single_element"),
  std::make_tuple(5, "two_elements_different_signs"),
  std::make_tuple(6, "two_elements_same_sign"),
  std::make_tuple(7, "long_alternating_sequence"),
  std::make_tuple(8, "mixed_sequence")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<NikitinATestTaskMPI, InType>(kTestParam, PPC_SETTINGS_nikitin_a_vec_sign_rotation),
                   ppc::util::AddFuncTask<NikitinATestTaskSEQ, InType>(kTestParam, PPC_SETTINGS_nikitin_a_vec_sign_rotation));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NikitinARunFuncTestsProcesses::PrintFuncTestName<NikitinARunFuncTestsProcesses>;

INSTANTIATE_TEST_SUITE_P(SignAlternationTests, NikitinARunFuncTestsProcesses, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nikitin_a_vec_sign_rotation