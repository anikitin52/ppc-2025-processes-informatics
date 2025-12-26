#pragma once

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "task/include/task.hpp"

namespace nikitin_a_fox_algorithm {

class NikitinAFoxAlgorithmMPI : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kMPI;
  }
  explicit NikitinAFoxAlgorithmMPI(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace nikitin_a_fox_algorithm
