#include "nikitin_a_fox_algorithm/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cmath>
#include <cstddef>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"

namespace nikitin_a_fox_algorithm {

NikitinAFoxAlgorithmMPI::NikitinAFoxAlgorithmMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool NikitinAFoxAlgorithmMPI::ValidateMatricesOnRoot(const std::vector<std::vector<double>> &matrix_a,
                                                     const std::vector<std::vector<double>> &matrix_b) {
  if (matrix_a.empty() || matrix_b.empty()) {
    return false;
  }

  const auto n = static_cast<int>(matrix_a.size());
  for (int i = 0; i < n; ++i) {
    if (matrix_a[i].size() != static_cast<std::size_t>(n)) {
      return false;
    }
  }

  if (matrix_b.size() != static_cast<std::size_t>(n)) {
    return false;
  }

  for (int i = 0; i < n; ++i) {
    if (matrix_b[i].size() != static_cast<std::size_t>(n)) {
      return false;
    }
  }

  return true;
}

bool NikitinAFoxAlgorithmMPI::ValidationImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    return ValidateMatricesOnRoot(GetInput().first, GetInput().second);
  }

  return true;
}

bool NikitinAFoxAlgorithmMPI::PreProcessingImpl() {
  return true;
}

void NikitinAFoxAlgorithmMPI::DistributeMatrixB(int n, const std::vector<std::vector<double>> &matrix_b,
                                                std::vector<double> &local_b) {
  for (int i = 0; i < n; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      local_b[i_offset + static_cast<std::size_t>(j)] = matrix_b[i][j];
    }
  }
}

void NikitinAFoxAlgorithmMPI::PrepareSendBuffer(int dest, int n, int rows_per_proc, int remainder,
                                                const std::vector<std::vector<double>> &matrix_a, int &current_row,
                                                std::vector<double> &send_buffer) {
  const int dest_rows = (dest < remainder) ? (rows_per_proc + 1) : rows_per_proc;

  for (int i = 0; i < dest_rows; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      send_buffer[i_offset + static_cast<std::size_t>(j)] = matrix_a[current_row][j];
    }
    current_row++;
  }
}

void NikitinAFoxAlgorithmMPI::PerformLocalMultiplication(int n, int local_rows, const std::vector<double> &local_a,
                                                         const std::vector<double> &local_b,
                                                         std::vector<double> &local_c) {
  for (int i = 0; i < local_rows; ++i) {
    const auto i_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        const auto k_idx = static_cast<std::size_t>(k) * static_cast<std::size_t>(n);
        sum += local_a[i_idx + static_cast<std::size_t>(k)] * local_b[k_idx + static_cast<std::size_t>(j)];
      }
      local_c[i_idx + static_cast<std::size_t>(j)] = sum;
    }
  }
}

void NikitinAFoxAlgorithmMPI::ProcessReceivedBlock(int n, const std::vector<double> &recv_buffer,
                                                   std::vector<std::vector<double>> &output, int &current_row) {
  const int src_rows = static_cast<int>(recv_buffer.size()) / n;

  for (int i = 0; i < src_rows; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      output[current_row][j] = recv_buffer[i_offset + static_cast<std::size_t>(j)];
    }
    current_row++;
  }
}

void NikitinAFoxAlgorithmMPI::FillFlatResult(int n, const std::vector<std::vector<double>> &matrix,
                                             std::vector<double> &flat_result) {
  for (int i = 0; i < n; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      flat_result[i_offset + static_cast<std::size_t>(j)] = matrix[i][j];
    }
  }
}

void NikitinAFoxAlgorithmMPI::FillFromFlatResult(int n, const std::vector<double> &flat_result,
                                                 std::vector<std::vector<double>> &matrix) {
  for (int i = 0; i < n; ++i) {
    const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
    for (int j = 0; j < n; ++j) {
      matrix[i][j] = flat_result[i_offset + static_cast<std::size_t>(j)];
    }
  }
}

bool NikitinAFoxAlgorithmMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Получаем размер матрицы
  int n = 0;
  if (rank == 0) {
    n = static_cast<int>(GetInput().first.size());
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    return false;
  }

  // Вычисляем распределение строк
  const int rows_per_proc = n / size;
  const int remainder = n % size;

  // Количество строк для текущего процесса
  const int local_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;
  const auto local_elements = static_cast<std::size_t>(local_rows) * static_cast<std::size_t>(n);

  // 1. Распределяем матрицу B всем процессам
  const auto total_elements = static_cast<std::size_t>(n) * static_cast<std::size_t>(n);
  std::vector<double> local_b(total_elements);

  if (rank == 0) {
    DistributeMatrixB(n, GetInput().second, local_b);
  }

  MPI_Bcast(local_b.data(), static_cast<int>(total_elements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // 2. Распределяем матрицу A по строкам
  std::vector<double> local_a(local_elements);

  if (rank == 0) {
    const auto &matrix_a = GetInput().first;

    // Сначала копируем строки для процесса 0
    int current_row = 0;
    for (int i = 0; i < local_rows; ++i) {
      const auto i_offset = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
      for (int j = 0; j < n; ++j) {
        local_a[i_offset + static_cast<std::size_t>(j)] = matrix_a[current_row][j];
      }
      current_row++;
    }

    // Отправляем строки остальным процессам
    for (int dest = 1; dest < size; ++dest) {
      const auto dest_elements = static_cast<std::size_t>((dest < remainder) ? (rows_per_proc + 1) : rows_per_proc) *
                                 static_cast<std::size_t>(n);
      std::vector<double> send_buffer(dest_elements);

      PrepareSendBuffer(dest, n, rows_per_proc, remainder, matrix_a, current_row, send_buffer);

      MPI_Send(send_buffer.data(), static_cast<int>(dest_elements), MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    }
  } else {
    MPI_Recv(local_a.data(), static_cast<int>(local_elements), MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // 3. Локальное умножение
  std::vector<double> local_c(local_elements, 0.0);
  PerformLocalMultiplication(n, local_rows, local_a, local_b, local_c);

  // 4. Сбор результатов на процессе 0
  if (rank == 0) {
    // Создаем результирующую матрицу
    GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    // Копируем свои результаты
    int current_row = 0;
    for (int i = 0; i < local_rows; ++i) {
      const auto i_idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(n);
      for (int j = 0; j < n; ++j) {
        GetOutput()[current_row][j] = local_c[i_idx + static_cast<std::size_t>(j)];
      }
      current_row++;
    }

    // Получаем результаты от других процессов
    for (int src = 1; src < size; ++src) {
      const auto src_elements = static_cast<std::size_t>((src < remainder) ? (rows_per_proc + 1) : rows_per_proc) *
                                static_cast<std::size_t>(n);
      std::vector<double> recv_buffer(src_elements);

      MPI_Recv(recv_buffer.data(), static_cast<int>(src_elements), MPI_DOUBLE, src, 1, MPI_COMM_WORLD,
               MPI_STATUS_IGNORE);

      ProcessReceivedBlock(n, recv_buffer, GetOutput(), current_row);
    }
  } else {
    // Отправляем свои результаты процессу 0
    MPI_Send(local_c.data(), static_cast<int>(local_elements), MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }

  // 5. Рассылаем результат всем процессам (для проверки в тестах)
  if (rank == 0) {
    // Преобразуем результат в плоский массив для рассылки
    std::vector<double> flat_result(total_elements);
    FillFlatResult(n, GetOutput(), flat_result);

    // Рассылаем плоский массив
    MPI_Bcast(flat_result.data(), static_cast<int>(total_elements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // На других процессах преобразуем обратно в матрицу
    FillFromFlatResult(n, flat_result, GetOutput());
  } else {
    // На других процессах создаем матрицу и получаем данные
    GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::vector<double> flat_result(total_elements);
    MPI_Bcast(flat_result.data(), static_cast<int>(total_elements), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    FillFromFlatResult(n, flat_result, GetOutput());
  }

  return true;
}

bool NikitinAFoxAlgorithmMPI::PostProcessingImpl() {
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &matrix_c = GetOutput();
  return !matrix_c.empty();
}

}  // namespace nikitin_a_fox_algorithm
