#pragma once

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_fox_algorithm {

class NikitinAFoxAlgorithmSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit NikitinAFoxAlgorithmSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

  // Вспомогательная функция
  void MultiplyBlocks(int n, int block_size, int grid_size, const std::vector<std::vector<double>> &matrix_a,
                      const std::vector<std::vector<double>> &matrix_b, std::vector<std::vector<double>> &matrix_c);
};

}  // namespace nikitin_a_fox_algorithm
