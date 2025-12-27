#pragma once

#include <vector>

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

  // Вспомогательные функции для уменьшения когнитивной сложности
  static bool ValidateMatricesOnRoot(const std::vector<std::vector<double>> &matrix_a,
                                     const std::vector<std::vector<double>> &matrix_b);
  static void DistributeMatrixB(int n, const std::vector<std::vector<double>> &matrix_b, std::vector<double> &local_b);
  static void PrepareSendBuffer(int dest, int n, int rows_per_proc, int remainder,
                                const std::vector<std::vector<double>> &matrix_a, int &current_row,
                                std::vector<double> &send_buffer);
  static void PerformLocalMultiplication(int n, int local_rows, const std::vector<double> &local_a,
                                         const std::vector<double> &local_b, std::vector<double> &local_c);
  static void ProcessReceivedBlock(int n, const std::vector<double> &recv_buffer,
                                   std::vector<std::vector<double>> &output, int &current_row);
  static void FillFlatResult(int n, const std::vector<std::vector<double>> &matrix, std::vector<double> &flat_result);
  static void FillFromFlatResult(int n, const std::vector<double> &flat_result,
                                 std::vector<std::vector<double>> &matrix);
};

}  // namespace nikitin_a_fox_algorithm
