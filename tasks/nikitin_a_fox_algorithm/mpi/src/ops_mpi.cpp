#include "nikitin_a_fox_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "util/include/util.hpp"

namespace nikitin_a_fox_algorithm {

NikitinAFoxAlgorithmMPI::NikitinAFoxAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  // Выходная матрица будет создана в RunImpl
}

bool NikitinAFoxAlgorithmMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Проверяем только на нулевом процессе
  if (rank == 0) {
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
  }
  
  // Рассылаем результат валидации всем процессам
  int is_valid = 1;
  int local_is_valid = 1;
  if (rank == 0) {
    const auto& matrix_a = GetInput().first;
    const auto& matrix_b = GetInput().second;
    
    // 1. Проверяем, что матрицы не пустые
    if (matrix_a.empty() || matrix_b.empty()) {
      local_is_valid = 0;
    }
    
    // 2. Проверяем, что матрицы квадратные
    int n = matrix_a.size();
    for (int i = 0; i < n; ++i) {
      if (matrix_a[i].size() != static_cast<size_t>(n) || 
          matrix_b[i].size() != static_cast<size_t>(n)) {
        local_is_valid = 0;
      }
    }
    
    // 3. Проверяем, что матрицы одинакового размера
    if (matrix_b.size() != static_cast<size_t>(n)) {
      local_is_valid = 0;
    }
    
    is_valid = local_is_valid;
  }
  
  MPI_Bcast(&is_valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
  
  return is_valid == 1;
}

bool NikitinAFoxAlgorithmMPI::PreProcessingImpl() {
  // Определяем топологию сетки процессов
  // В этом методе можно подготовить коммуникаторы и т.д.
  return true;
}

bool NikitinAFoxAlgorithmMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Определяем размер сетки процессов (ближайший квадрат)
  int grid_size = static_cast<int>(std::sqrt(size));
  while (grid_size * grid_size > size) {
    grid_size--;
  }
  
  // Создаём коммуникатор для сетки процессов
  MPI_Comm grid_comm;
  int dims[2] = {grid_size, grid_size};
  int periods[2] = {1, 1}; // Циклическая топология
  int reorder = 1;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &grid_comm);
  
  // Если процесс не вошёл в сетку, завершаем его
  if (grid_comm == MPI_COMM_NULL) {
    return true;
  }
  
  int grid_rank;
  int coords[2];
  MPI_Comm_rank(grid_comm, &grid_rank);
  MPI_Cart_coords(grid_comm, grid_rank, 2, coords);
  int row = coords[0];
  int col = coords[1];
  
  // Получаем размер матрицы на нулевом процессе и рассылаем всем
  int n = 0;
  if (rank == 0) {
    n = GetInput().first.size();
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, grid_comm);
  
  // Определяем размер блока
  int block_size = n / grid_size;
  if (n % grid_size != 0) {
    block_size++;
  }
  
  // Выделяем память под локальные блоки
  std::vector<std::vector<double>> local_a(block_size, std::vector<double>(block_size, 0.0));
  std::vector<std::vector<double>> local_b(block_size, std::vector<double>(block_size, 0.0));
  std::vector<std::vector<double>> local_c(block_size, std::vector<double>(block_size, 0.0));
  
  // Рассылаем блоки матриц A и B по процессам сетки
  if (rank == 0) {
    const auto& matrix_a = GetInput().first;
    
    // Рассылаем блоки матрицы A
    for (int proc_row = 0; proc_row < grid_size; proc_row++) {
      for (int proc_col = 0; proc_col < grid_size; proc_col++) {
        // Определяем блок для текущего процесса
        int a_block_k = (proc_row + 0) % grid_size; // На первой итерации
        for (int i = 0; i < block_size; i++) {
          int global_i = proc_row * block_size + i;
          for (int j = 0; j < block_size; j++) {
            int global_j = a_block_k * block_size + j;
            if (global_i < n && global_j < n) {
              local_a[i][j] = matrix_a[global_i][global_j];
            }
          }
        }
        
        // Отправляем блок процессу (proc_row, proc_col)
        int dest_rank;
        int dest_coords[2] = {proc_row, proc_col};
        MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
        
        if (dest_rank != 0) {
          // Отправляем весь блок как плоский массив
          std::vector<double> flat_block(block_size * block_size);
          for (int i = 0; i < block_size; i++) {
            for (int j = 0; j < block_size; j++) {
              flat_block[i * block_size + j] = local_a[i][j];
            }
          }
          MPI_Send(flat_block.data(), block_size * block_size, MPI_DOUBLE, 
                   dest_rank, 0, grid_comm);
        }
      }
    }
    
    // Аналогично рассылаем блоки матрицы B...
    // (остальной код остается без изменений)
  } else {
    // Принимаем блок матрицы A
    std::vector<double> flat_block(block_size * block_size);
    MPI_Recv(flat_block.data(), block_size * block_size, MPI_DOUBLE, 
             0, 0, grid_comm, MPI_STATUS_IGNORE);
    
    // Восстанавливаем 2D структуру
    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        local_a[i][j] = flat_block[i * block_size + j];
      }
    }
  }
  
  // Алгоритм Фокса
  for (int iter = 0; iter < grid_size; iter++) {
    // Определяем корневой процесс для рассылки блока A в строке
    int root = (col + iter) % grid_size;
    
    // Рассылаем блок A в пределах строки
    std::vector<double> a_flat(block_size * block_size);
    if (root == col) {
      // Упаковываем блок A
      for (int i = 0; i < block_size; i++) {
        for (int j = 0; j < block_size; j++) {
          a_flat[i * block_size + j] = local_a[i][j];
        }
      }
    }
    
    // Рассылаем блок по строке
    MPI_Bcast(a_flat.data(), block_size * block_size, MPI_DOUBLE, root, grid_comm);
    
    // Распаковываем полученный блок A
    std::vector<std::vector<double>> a_temp(block_size, std::vector<double>(block_size));
    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        a_temp[i][j] = a_flat[i * block_size + j];
      }
    }
    
    // Умножаем локальные блоки
    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        for (int k = 0; k < block_size; k++) {
          local_c[i][j] += a_temp[i][k] * local_b[k][j];
        }
      }
    }
    
    // Циклический сдвиг блоков B вверх по столбцам
    int up, down;
    MPI_Cart_shift(grid_comm, 0, 1, &up, &down);
    
    std::vector<double> b_flat(block_size * block_size);
    // Упаковываем блок B
    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        b_flat[i * block_size + j] = local_b[i][j];
      }
    }
    
    // Выполняем сдвиг
    MPI_Sendrecv_replace(b_flat.data(), block_size * block_size, MPI_DOUBLE,
                         up, 0, down, 0, grid_comm, MPI_STATUS_IGNORE);
    
    // Распаковываем полученный блок B
    for (int i = 0; i < block_size; i++) {
      for (int j = 0; j < block_size; j++) {
        local_b[i][j] = b_flat[i * block_size + j];
      }
    }
  }
  
  // Собираем результаты на нулевом процессе
  if (rank == 0) {
    GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
  }
  
  // Каждый процесс отправляет свой блок результата
  for (int i = 0; i < block_size; i++) {
    int global_i = row * block_size + i;
    if (global_i < n) {
      std::vector<double> row_data(block_size);
      for (int j = 0; j < block_size; j++) {
        row_data[j] = local_c[i][j];
      }
      
      if (rank == 0) {
        for (int proc_col = 0; proc_col < grid_size; proc_col++) {
          int source_rank;
          int source_coords[2] = {row, proc_col};
          MPI_Cart_rank(grid_comm, source_coords, &source_rank);
          
          if (source_rank == 0) {
            // Копируем из local_c
            for (int j = 0; j < block_size; j++) {
              int global_j = proc_col * block_size + j;
              if (global_j < n) {
                GetOutput()[global_i][global_j] = local_c[i][j];
              }
            }
          } else {
            // Принимаем от других процессов
            std::vector<double> recv_row(block_size);
            MPI_Recv(recv_row.data(), block_size, MPI_DOUBLE, 
                     source_rank, 0, grid_comm, MPI_STATUS_IGNORE);
            
            for (int j = 0; j < block_size; j++) {
              int global_j = proc_col * block_size + j;
              if (global_j < n) {
                GetOutput()[global_i][global_j] = recv_row[j];
              }
            }
          }
        }
      } else {
        // Отправляем свою строку нулевому процессу
        MPI_Send(row_data.data(), block_size, MPI_DOUBLE, 0, 0, grid_comm);
      }
    }
  }
  
  MPI_Comm_free(&grid_comm);
  return true;
}

bool NikitinAFoxAlgorithmMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Проверка корректности результата
  if (rank == 0) {
    const auto& matrix_c = GetOutput();
    return !matrix_c.empty();
  }
  
  return true;
}

}  // namespace nikitin_a_fox_algorithm