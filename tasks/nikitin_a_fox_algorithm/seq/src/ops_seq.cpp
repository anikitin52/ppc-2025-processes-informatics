#include "nikitin_a_fox_algorithm/seq/include/ops_seq.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace nikitin_a_fox_algorithm {

NikitinAFoxAlgorithmSEQ::NikitinAFoxAlgorithmSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  // Выходная матрица будет создана в RunImpl после проверки размеров
}

bool NikitinAFoxAlgorithmSEQ::ValidationImpl() {
  // Проверка корректности полученных данных
  const auto& matrix_a = GetInput().first;
  const auto& matrix_b = GetInput().second;
  
  // 1. Проверяем, что матрицы не пустые
  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }
  
  // 2. Проверяем, что матрицы квадратные
  int n = matrix_a.size();
  for (int i = 0; i < n; ++i) {
    if (matrix_a[i].size() != static_cast<size_t>(n) || 
        matrix_b[i].size() != static_cast<size_t>(n)) {
      return false;
    }
  }
  
  // 3. Проверяем, что матрицы одинакового размера
  if (matrix_b.size() != static_cast<size_t>(n)) {
    return false;
  }
  
  return true;
}

bool NikitinAFoxAlgorithmSEQ::PreProcessingImpl() {
  // Подготовка данных - проверяем размеры для блочного разбиения
  // В последовательной версии просто убеждаемся, что данные валидны
  return true;
}

bool NikitinAFoxAlgorithmSEQ::RunImpl() {
  const auto& [matrix_a, matrix_b] = GetInput();
  
  int n = matrix_a.size();
  
  // Инициализируем выходную матрицу нулями
  std::vector<std::vector<double>> matrix_c(n, std::vector<double>(n, 0.0));
  
  // Определяем размер блока - для последовательной версии выбираем оптимальный
  // Можно использовать размер блока = sqrt(n) или фиксированное значение
  int block_size = 64; // Размер блока для кэш-оптимизации
  if (n < block_size) {
    block_size = n;
  }
  
  // Вычисляем количество блоков
  int grid_size = (n + block_size - 1) / block_size; // ceil(n/block_size)
  
  // Последовательная реализация алгоритма Фокса
  for (int iter = 0; iter < grid_size; ++iter) {
    for (int block_i = 0; block_i < grid_size; ++block_i) {
      for (int block_j = 0; block_j < grid_size; ++block_j) {
        // Вычисляем, какой блок матрицы A "активен" на этой итерации
        int a_block_k = (block_i + iter) % grid_size;
        
        // Границы текущих блоков
        int a_row_start = block_i * block_size;
        int a_row_end = std::min(a_row_start + block_size, n);
        int a_col_start = a_block_k * block_size;
        int a_col_end = std::min(a_col_start + block_size, n);
        
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
  // Проверка корректности результата
  // Можно добавить проверку размеров или других инвариантов
  const auto& matrix_c = GetOutput();
  return !matrix_c.empty();
}

}  // namespace nikitin_a_fox_algorithm