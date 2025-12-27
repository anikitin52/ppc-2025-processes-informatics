#include "nikitin_a_fox_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"

namespace nikitin_a_fox_algorithm {

NikitinAFoxAlgorithmSEQ::NikitinAFoxAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool NikitinAFoxAlgorithmSEQ::ValidationImpl() {
  const auto &matrix_a = GetInput().first;
  const auto &matrix_b = GetInput().second;

  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  auto n = static_cast<int>(matrix_a.size());
  for (int i = 0; i < n; ++i) {
    if (matrix_a[i].size() != static_cast<size_t>(n)) {
      return false;
    }
  }

  if (matrix_b.size() != static_cast<size_t>(n)) {
    return false;
  }

  for (int i = 0; i < n; ++i) {
    if (matrix_b[i].size() != static_cast<size_t>(n)) {
      return false;
    }
  }

  return true;
}

bool NikitinAFoxAlgorithmSEQ::PreProcessingImpl() {
  return true;
}

bool NikitinAFoxAlgorithmSEQ::RunImpl() {
  const auto &[matrix_a, matrix_b] = GetInput();

  auto n = static_cast<int>(matrix_a.size());

  // Инициализируем выходную матрицу нулями
  std::vector<std::vector<double>> matrix_c(n, std::vector<double>(n, 0.0));

  // Определяем размер блока - выбираем оптимальный для кэша
  int block_size = 64;
  block_size = std::min(n, block_size);

  // Вычисляем количество блоков
  int grid_size = (n + block_size - 1) / block_size;

  // Алгоритм Фокса (последовательная версия)
  for (int iter = 0; iter < grid_size; ++iter) {
    for (int block_i = 0; block_i < grid_size; ++block_i) {
      for (int block_j = 0; block_j < grid_size; ++block_j) {
        // Вычисляем, какой блок матрицы A "активен" на этой итерации
        int a_block_k = (block_i + iter) % grid_size;

        // Границы блоков для A
        int a_row_start = block_i * block_size;
        int a_row_end = std::min(a_row_start + block_size, n);
        int a_col_start = a_block_k * block_size;
        int a_col_end = std::min(a_col_start + block_size, n);

        // Границы блоков для B и C
        int b_col_start = block_j * block_size;
        int b_col_end = std::min(b_col_start + block_size, n);

        // Умножаем блоки матриц
        for (int i = a_row_start; i < a_row_end; ++i) {
          for (int k = a_col_start; k < a_col_end; ++k) {
            double a_ik = matrix_a[i][k];
            for (int j = b_col_start; j < b_col_end; ++j) {
              matrix_c[i][j] += a_ik * matrix_b[k][j];
            }
          }
        }
      }
    }
  }

  // Сохраняем результат
  GetOutput() = matrix_c;

  return true;
}

bool NikitinAFoxAlgorithmSEQ::PostProcessingImpl() {
  const auto &matrix_c = GetOutput();
  return !matrix_c.empty();
}

}  // namespace nikitin_a_fox_algorithm
