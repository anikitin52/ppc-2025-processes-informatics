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

  // Вспомогательные функции
  bool ValidateMatricesOnRoot();
  void DistributeMatrixB(int n, std::vector<double> &local_b);
  void DistributeMatrixA(int rank, int size, int n, int local_rows, std::vector<double> &local_a);
  void PerformLocalMultiplication(int n, int local_rows, const std::vector<double> &local_a,
                                  const std::vector<double> &local_b, std::vector<double> &local_c);
  void CollectResults(int rank, int size, int n, int local_rows, const std::vector<double> &local_c);
  void BroadcastResults(int rank, int n);
};

}  // namespace nikitin_a_fox_algorithm
