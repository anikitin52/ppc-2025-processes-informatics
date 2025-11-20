#include <gtest/gtest.h>

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"
#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace nikitin_a_vec_sign_rotation {

class NikitinARunPerfTestProcesses : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    // Загружаем данные из файла
    std::ifstream file("tasks/nikitin_a_vec_sign_rotation/data/large_vector.txt");
    std::string line;
    
    if (std::getline(file, line)) {
      std::stringstream ss(line);
      std::string token;
      
      while (std::getline(ss, token, ',')) {
        // Убираем пробелы
        token.erase(0, token.find_first_not_of(' '));
        token.erase(token.find_last_not_of(' ') + 1);
        input_data_.push_back(std::stoi(token));
      }
    }
    
    // Проверяем что загрузили достаточно данных
    if (input_data_.size() < 1000000) {
      std::cout << "WARNING: Loaded only " << input_data_.size() << " elements" << std::endl;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Для performance тестов проверяем разумность результата
    return (output_data >= 0) && (output_data <= static_cast<OutType>(input_data_.size() - 1));
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_;
};

TEST_P(NikitinARunPerfTestProcesses, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, NikitinATestTaskMPI, NikitinATestTaskSEQ>(PPC_SETTINGS_nikitin_a_vec_sign_rotation);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = NikitinARunPerfTestProcesses::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, NikitinARunPerfTestProcesses, kGtestValues, kPerfTestName);

}  // namespace nikitin_a_vec_sign_rotation