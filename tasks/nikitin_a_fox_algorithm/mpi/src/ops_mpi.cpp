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
}

bool NikitinAFoxAlgorithmMPI::ValidationImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    const auto& matrix_a = GetInput().first;
    const auto& matrix_b = GetInput().second;
    
    if (matrix_a.empty() || matrix_b.empty()) {
      return false;
    }
    
    int n = matrix_a.size();
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
  }
  
  return true;
}

bool NikitinAFoxAlgorithmMPI::PreProcessingImpl() {
  return true;
}

bool NikitinAFoxAlgorithmMPI::RunImpl() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  
  // Определяем размер сетки процессов
  int grid_size = static_cast<int>(std::sqrt(size));
  while (grid_size * grid_size > size) {
    grid_size--;
  }
  
  // Если процессов недостаточно для сетки, используем последовательную версию
  if (grid_size == 0) {
    // Простая последовательная реализация для случая одного процесса
    if (rank == 0) {
      const auto& matrix_a = GetInput().first;
      const auto& matrix_b = GetInput().second;
      int n = matrix_a.size();
      
      std::vector<std::vector<double>> matrix_c(n, std::vector<double>(n, 0.0));
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          for (int k = 0; k < n; ++k) {
            matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
          }
        }
      }
      GetOutput() = matrix_c;
    }
    return true;
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
  
  // Получаем размер матрицы
  int n = 0;
  if (rank == 0) {
    n = GetInput().first.size();
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, grid_comm);
  
  // Определяем размер блока для каждого процесса
  int block_rows = (n + grid_size - 1) / grid_size;  // ceil(n/grid_size)
  int block_cols = block_rows;
  
  // Выделяем память под локальные блоки с учетом padding для неравномерного распределения
  std::vector<std::vector<double>> local_a(block_rows, std::vector<double>(block_cols, 0.0));
  std::vector<std::vector<double>> local_b(block_rows, std::vector<double>(block_cols, 0.0));
  std::vector<std::vector<double>> local_c(block_rows, std::vector<double>(block_cols, 0.0));
  
  // Распределяем матрицу A по процессам
  if (rank == 0) {
    const auto& matrix_a = GetInput().first;
    const auto& matrix_b = GetInput().second;
    
    // Рассылаем блоки матрицы A
    std::vector<double> send_buffer(block_rows * block_cols);
    for (int proc_row = 0; proc_row < grid_size; proc_row++) {
      for (int proc_col = 0; proc_col < grid_size; proc_col++) {
        // Заполняем буфер данными для процесса (proc_row, proc_col)
        for (int i = 0; i < block_rows; i++) {
          int global_i = proc_row * block_rows + i;
          for (int j = 0; j < block_cols; j++) {
            int global_j = proc_col * block_cols + j;
            if (global_i < n && global_j < n) {
              send_buffer[i * block_cols + j] = matrix_a[global_i][global_j];
            } else {
              send_buffer[i * block_cols + j] = 0.0;
            }
          }
        }
        
        // Отправляем данные процессу
        int dest_rank;
        int dest_coords[2] = {proc_row, proc_col};
        MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
        
        if (dest_rank == 0) {
          // Копируем данные для процесса 0
          for (int i = 0; i < block_rows; i++) {
            for (int j = 0; j < block_cols; j++) {
              local_a[i][j] = send_buffer[i * block_cols + j];
            }
          }
        } else {
          MPI_Send(send_buffer.data(), block_rows * block_cols, MPI_DOUBLE, 
                   dest_rank, 0, grid_comm);
        }
      }
    }
    
    // Рассылаем блоки матрицы B
    for (int proc_row = 0; proc_row < grid_size; proc_row++) {
      for (int proc_col = 0; proc_col < grid_size; proc_col++) {
        // Заполняем буфер данными для процесса (proc_row, proc_col)
        for (int i = 0; i < block_rows; i++) {
          int global_i = proc_row * block_rows + i;
          for (int j = 0; j < block_cols; j++) {
            int global_j = proc_col * block_cols + j;
            if (global_i < n && global_j < n) {
              send_buffer[i * block_cols + j] = matrix_b[global_i][global_j];
            } else {
              send_buffer[i * block_cols + j] = 0.0;
            }
          }
        }
        
        // Отправляем данные процессу
        int dest_rank;
        int dest_coords[2] = {proc_row, proc_col};
        MPI_Cart_rank(grid_comm, dest_coords, &dest_rank);
        
        if (dest_rank == 0) {
          // Копируем данные для процесса 0
          for (int i = 0; i < block_rows; i++) {
            for (int j = 0; j < block_cols; j++) {
              local_b[i][j] = send_buffer[i * block_cols + j];
            }
          }
        } else {
          MPI_Send(send_buffer.data(), block_rows * block_cols, MPI_DOUBLE, 
                   dest_rank, 1, grid_comm);
        }
      }
    }
  } else {
    // Принимаем блок матрицы A
    std::vector<double> recv_buffer(block_rows * block_cols);
    MPI_Recv(recv_buffer.data(), block_rows * block_cols, MPI_DOUBLE, 
             0, 0, grid_comm, MPI_STATUS_IGNORE);
    
    for (int i = 0; i < block_rows; i++) {
      for (int j = 0; j < block_cols; j++) {
        local_a[i][j] = recv_buffer[i * block_cols + j];
      }
    }
    
    // Принимаем блок матрицы B
    MPI_Recv(recv_buffer.data(), block_rows * block_cols, MPI_DOUBLE, 
             0, 1, grid_comm, MPI_STATUS_IGNORE);
    
    for (int i = 0; i < block_rows; i++) {
      for (int j = 0; j < block_cols; j++) {
        local_b[i][j] = recv_buffer[i * block_cols + j];
      }
    }
  }
  
  // Алгоритм Фокса
  // Инициализируем локальную матрицу C нулями
  for (int i = 0; i < block_rows; i++) {
    std::fill(local_c[i].begin(), local_c[i].end(), 0.0);
  }
  
  // Определяем коммуникаторы для строк и столбцов
  MPI_Comm row_comm, col_comm;
  int remain_dims_row[2] = {0, 1}; // Сохраняем столбцы
  int remain_dims_col[2] = {1, 0}; // Сохраняем строки
  
  MPI_Cart_sub(grid_comm, remain_dims_row, &row_comm);
  MPI_Cart_sub(grid_comm, remain_dims_col, &col_comm);
  
  // Создаем временную матрицу для блока A
  std::vector<double> a_temp(block_rows * block_cols, 0.0);
  std::vector<double> a_buffer(block_rows * block_cols, 0.0);
  
  for (int iter = 0; iter < grid_size; iter++) {
    // Определяем корневой процесс для текущей итерации в строке
    int root = (col + iter) % grid_size;
    
    // Рассылаем блок A в пределах строки
    if (col == root) {
      // Упаковываем блок A
      for (int i = 0; i < block_rows; i++) {
        for (int j = 0; j < block_cols; j++) {
          a_buffer[i * block_cols + j] = local_a[i][j];
        }
      }
    }
    
    // Рассылаем блок по строке
    MPI_Bcast(a_buffer.data(), block_rows * block_cols, MPI_DOUBLE, root, row_comm);
    
    // Распаковываем полученный блок A
    for (int i = 0; i < block_rows; i++) {
      for (int j = 0; j < block_cols; j++) {
        a_temp[i * block_cols + j] = a_buffer[i * block_cols + j];
      }
    }
    
    // Умножаем локальные блоки (a_temp * local_b)
    for (int i = 0; i < block_rows; i++) {
      for (int j = 0; j < block_cols; j++) {
        double sum = 0.0;
        for (int k = 0; k < block_cols; k++) {
          sum += a_temp[i * block_cols + k] * local_b[k][j];
        }
        local_c[i][j] += sum;
      }
    }
    
    // Циклический сдвиг блоков B вверх по столбцам
    std::vector<double> b_buffer(block_rows * block_cols);
    // Упаковываем блок B
    for (int i = 0; i < block_rows; i++) {
      for (int j = 0; j < block_cols; j++) {
        b_buffer[i * block_cols + j] = local_b[i][j];
      }
    }
    
    // Выполняем сдвиг
    MPI_Status status;
    int up, down;
    MPI_Cart_shift(col_comm, 0, 1, &up, &down);
    
    MPI_Sendrecv_replace(b_buffer.data(), block_rows * block_cols, MPI_DOUBLE,
                         up, 0, down, 0, col_comm, &status);
    
    // Распаковываем полученный блок B
    for (int i = 0; i < block_rows; i++) {
      for (int j = 0; j < block_cols; j++) {
        local_b[i][j] = b_buffer[i * block_cols + j];
      }
    }
  }
  
  // Собираем результаты на нулевом процессе
  if (rank == 0) {
    GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
  }
  
  // Каждый процесс отправляет свой блок результата
  for (int i = 0; i < block_rows; i++) {
    int global_i = row * block_rows + i;
    if (global_i < n) {
      std::vector<double> row_buffer(block_cols);
      for (int j = 0; j < block_cols; j++) {
        row_buffer[j] = local_c[i][j];
      }
      
      // Отправляем строку блока на процесс 0
      if (grid_rank != 0) {
        MPI_Send(row_buffer.data(), block_cols, MPI_DOUBLE, 0, 0, grid_comm);
      } else {
        // Процесс 0 собирает данные от других процессов
        for (int proc_col = 0; proc_col < grid_size; proc_col++) {
          std::vector<double> recv_row(block_cols);
          
          if (proc_col == col) {
            // Используем локальные данные
            recv_row = row_buffer;
          } else {
            // Получаем данные от другого процесса в этой строке
            int source_rank;
            int source_coords[2] = {row, proc_col};
            MPI_Cart_rank(grid_comm, source_coords, &source_rank);
            MPI_Recv(recv_row.data(), block_cols, MPI_DOUBLE, 
                     source_rank, 0, grid_comm, MPI_STATUS_IGNORE);
          }
          
          // Копируем полученные данные в итоговую матрицу
          for (int j = 0; j < block_cols; j++) {
            int global_j = proc_col * block_cols + j;
            if (global_j < n) {
              GetOutput()[global_i][global_j] = recv_row[j];
            }
          }
        }
      }
    }
  }
  
  // Если мы не процесс 0, отправляем все строки
  if (grid_rank != 0) {
    for (int i = 0; i < block_rows; i++) {
      int global_i = row * block_rows + i;
      if (global_i < n) {
        std::vector<double> row_buffer(block_cols);
        for (int j = 0; j < block_cols; j++) {
          row_buffer[j] = local_c[i][j];
        }
        MPI_Send(row_buffer.data(), block_cols, MPI_DOUBLE, 0, 0, grid_comm);
      }
    }
  }
  
  // Освобождаем коммуникаторы
  MPI_Comm_free(&row_comm);
  MPI_Comm_free(&col_comm);
  MPI_Comm_free(&grid_comm);
  
  return true;
}

bool NikitinAFoxAlgorithmMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (rank == 0) {
    const auto& matrix_c = GetOutput();
    return !matrix_c.empty();
  }
  
  return true;
}

}  // namespace nikitin_a_fox_algorithm