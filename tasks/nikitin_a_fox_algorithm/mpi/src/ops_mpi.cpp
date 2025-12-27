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
    const auto &matrix_a = GetInput().first;
    const auto &matrix_b = GetInput().second;

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

  // Получаем размер матрицы
  int n = 0;
  if (rank == 0) {
    n = GetInput().first.size();
  }
  MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (n == 0) {
    return false;
  }

  // Вычисляем распределение строк
  int rows_per_proc = n / size;
  int remainder = n % size;

  // Количество строк для текущего процесса
  int local_rows = (rank < remainder) ? (rows_per_proc + 1) : rows_per_proc;
  int local_elements = local_rows * n;

  // Создаем массив для матрицы B (будет одинаковый на всех процессах)
  std::vector<double> local_b(n * n);

  // 1. Распределяем матрицу B всем процессам
  if (rank == 0) {
    const auto &matrix_b = GetInput().second;
    // Заполняем локальный буфер матрицей B
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        local_b[i * n + j] = matrix_b[i][j];
      }
    }
  }

  // Рассылаем матрицу B всем процессам
  MPI_Bcast(local_b.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  // 2. Распределяем матрицу A по строкам
  std::vector<double> local_a(local_elements);

  if (rank == 0) {
    const auto &matrix_a = GetInput().first;

    // Сначала копируем строки для процесса 0
    int current_row = 0;
    for (int i = 0; i < local_rows; ++i) {
      for (int j = 0; j < n; ++j) {
        local_a[i * n + j] = matrix_a[current_row][j];
      }
      current_row++;
    }

    // Отправляем строки остальным процессам
    for (int dest = 1; dest < size; ++dest) {
      int dest_rows = (dest < remainder) ? (rows_per_proc + 1) : rows_per_proc;
      std::vector<double> send_buffer(dest_rows * n);

      for (int i = 0; i < dest_rows; ++i) {
        for (int j = 0; j < n; ++j) {
          send_buffer[i * n + j] = matrix_a[current_row][j];
        }
        current_row++;
      }

      MPI_Send(send_buffer.data(), dest_rows * n, MPI_DOUBLE, dest, 0, MPI_COMM_WORLD);
    }
  } else {
    // Получаем свои строки от процесса 0
    MPI_Recv(local_a.data(), local_elements, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // 3. Локальное умножение
  std::vector<double> local_c(local_elements, 0.0);

  for (int i = 0; i < local_rows; ++i) {
    for (int j = 0; j < n; ++j) {
      double sum = 0.0;
      for (int k = 0; k < n; ++k) {
        sum += local_a[i * n + k] * local_b[k * n + j];
      }
      local_c[i * n + j] = sum;
    }
  }

  // 4. Сбор результатов на процессе 0
  if (rank == 0) {
    // Создаем результирующую матрицу
    GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));

    // Копируем свои результаты
    int current_row = 0;
    for (int i = 0; i < local_rows; ++i) {
      for (int j = 0; j < n; ++j) {
        GetOutput()[current_row][j] = local_c[i * n + j];
      }
      current_row++;
    }

    // Получаем результаты от других процессов
    for (int src = 1; src < size; ++src) {
      int src_rows = (src < remainder) ? (rows_per_proc + 1) : rows_per_proc;
      std::vector<double> recv_buffer(src_rows * n);

      MPI_Recv(recv_buffer.data(), src_rows * n, MPI_DOUBLE, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      for (int i = 0; i < src_rows; ++i) {
        for (int j = 0; j < n; ++j) {
          GetOutput()[current_row][j] = recv_buffer[i * n + j];
        }
        current_row++;
      }
    }
  } else {
    // Отправляем свои результаты процессу 0
    MPI_Send(local_c.data(), local_elements, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD);
  }

  // 5. Рассылаем результат всем процессам (для проверки в тестах)
  if (rank == 0) {
    // Преобразуем результат в плоский массив для рассылки
    std::vector<double> flat_result(n * n);
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        flat_result[i * n + j] = GetOutput()[i][j];
      }
    }

    // Рассылаем плоский массив
    MPI_Bcast(flat_result.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // На других процессах преобразуем обратно в матрицу
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        GetOutput()[i][j] = flat_result[i * n + j];
      }
    }
  } else {
    // На других процессах создаем матрицу и получаем данные
    GetOutput() = std::vector<std::vector<double>>(n, std::vector<double>(n));

    std::vector<double> flat_result(n * n);
    MPI_Bcast(flat_result.data(), n * n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        GetOutput()[i][j] = flat_result[i * n + j];
      }
    }
  }

  return true;
}

bool NikitinAFoxAlgorithmMPI::PostProcessingImpl() {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  const auto &matrix_c = GetOutput();
  return !matrix_c.empty();
}

}  // namespace nikitin_a_fox_algorithm
