#include "nikitin_a_vec_sign_rotation/mpi/include/ops_mpi.hpp"

#include <mpi.h>
#include <vector>

namespace nikitin_a_vec_sign_rotation {

NikitinATestTaskMPI::NikitinATestTaskMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NikitinATestTaskMPI::ValidationImpl() {
  const auto& vector_data = GetInput();
  return !vector_data.empty();
}

bool NikitinATestTaskMPI::PreProcessingImpl() {
  return true;
}

bool NikitinATestTaskMPI::RunImpl() {
  const auto& full_vector = GetInput();
  
  int world_size, world_rank;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  // Если процессов больше или равно количеству элементов - используем SEQ логику
  if (full_vector.size() <= static_cast<size_t>(world_size)) {
    if (world_rank == 0) {
      // SEQ логика для маленьких векторов
      if (full_vector.size() < 2) {
        GetOutput() = 0;
      } else {
        int sign_alternations = 0;
        for (size_t j = 1; j < full_vector.size(); j++) {
          if ((full_vector[j - 1] < 0 && full_vector[j] >= 0) || 
              (full_vector[j - 1] >= 0 && full_vector[j] < 0)) {
            sign_alternations++;
          }
        }
        GetOutput() = sign_alternations;
      }
    } else {
      // Остальные процессы ничего не делают
      GetOutput() = 0;
    }
    return true;
  }


  // Распределяем данные между процессами
  size_t total_size = full_vector.size();
  size_t elements_per_process = total_size / world_size;
  size_t remainder = total_size % world_size;

  // Определяем границы для каждого процесса
  size_t local_start, local_end;
  
  if (world_rank < remainder) {
    local_start = world_rank * (elements_per_process + 1);
    local_end = local_start + elements_per_process + 1;
  } else {
    local_start = world_rank * elements_per_process + remainder;
    local_end = local_start + elements_per_process;
  }

  // Корректируем границы для учета перекрытий
  size_t actual_local_start = local_start;
  size_t actual_local_end = local_end;
  
  // Каждый процесс (кроме первого) начинает на 1 элемент раньше
  if (world_rank > 0) {
    actual_local_start = local_start - 1;
  }
  
  // Каждый процесс (кроме последнего) заканчивает на 1 элемент позже
  if (world_rank < world_size - 1) {
    actual_local_end = local_end + 1;
  }

  // Создаем локальный вектор для каждого процесса
  std::vector<int> local_vector;
  if (world_rank == 0) {
    // Процесс 0 распределяет данные
    for (int proc = 0; proc < world_size; proc++) {
      size_t proc_start, proc_end;
      
      if (proc < remainder) {
        proc_start = proc * (elements_per_process + 1);
        proc_end = proc_start + elements_per_process + 1;
      } else {
        proc_start = proc * elements_per_process + remainder;
        proc_end = proc_start + elements_per_process;
      }
      
      // Корректируем границы для учета перекрытий
      if (proc > 0) {
        proc_start = proc_start - 1;
      }
      if (proc < world_size - 1) {
        proc_end = proc_end + 1;
      }
      
      size_t proc_size = proc_end - proc_start;
      std::vector<int> proc_data(proc_size);
      
      for (size_t i = 0; i < proc_size; i++) {
        proc_data[i] = full_vector[proc_start + i];
      }
      
      if (proc == 0) {
        local_vector = proc_data;
      } else {
        MPI_Send(proc_data.data(), proc_size, MPI_INT, proc, 0, MPI_COMM_WORLD);
      }
    }
  } else {
    // Остальные процессы получают данные
    size_t local_size = actual_local_end - actual_local_start;
    local_vector.resize(local_size);
    MPI_Recv(local_vector.data(), local_size, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Локальный подсчет чередований
  int local_alternations = 0;
  
  // Для корректного учета границ:
  size_t start_index = (world_rank == 0) ? 1 : 2;
  
  for (size_t j = start_index; j < local_vector.size(); j++) {
    if ((local_vector[j - 1] < 0 && local_vector[j] >= 0) || 
        (local_vector[j - 1] >= 0 && local_vector[j] < 0)) {
      local_alternations++;
    }
  }

  // Собираем результаты
  int total_alternations = 0;
  MPI_Reduce(&local_alternations, &total_alternations, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  if (world_rank == 0) {
    GetOutput() = total_alternations;
  }

  return true;
}

bool NikitinATestTaskMPI::PostProcessingImpl() {
  return true;
}

}  // namespace nikitin_a_vec_sign_rotation