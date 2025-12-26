#include <gtest/gtest.h>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "nikitin_a_fox_algorithm/mpi/include/ops_mpi.hpp"
#include "nikitin_a_fox_algorithm/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_fox_algorithm {

class NikitinAFoxAlgorithmPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(NikitinAFoxAlgorithmPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikitinAFoxAlgorithmMPI, NikitinAFoxAlgorithmSEQ>(PPC_SETTINGS_nikitin_a_fox_algorithm);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinAFoxAlgorithmPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinAFoxAlgorithmPerfTests, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_fox_algorithm
