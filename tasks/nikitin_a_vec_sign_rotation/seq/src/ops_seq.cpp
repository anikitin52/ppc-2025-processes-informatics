#include "nikitin_a_vec_sign_rotation/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "nikitin_a_vec_sign_rotation/common/include/common.hpp"
#include "util/include/util.hpp"

namespace nikitin_a_vec_sign_rotation {

NikitinATestTaskSEQ::NikitinATestTaskSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool NikitinATestTaskSEQ::ValidationImpl() {
  // Vector is not empty
  const auto& vector_data = GetInput();
  return !vector_data.empty();
}

bool NikitinATestTaskSEQ::PreProcessingImpl() {
  return true;
}

bool NikitinATestTaskSEQ::RunImpl() {
  const auto& vect_data = GetInput();
  
  // Если вектор слишком маленький, чередований нет
  if (vect_data.size() < 2) {
    GetOutput() = 0;
    return true;
  }

  int sign_alternations = 0;
  
  // Подсчитываем чередования знаков между соседними элементами
  for (size_t j = 1; j < vect_data.size(); j++) {
    // Проверяем, что знаки текущего и предыдущего элемента разные
    if ((vect_data[j - 1] < 0 && vect_data[j] >= 0) || 
        (vect_data[j - 1] >= 0 && vect_data[j] < 0)) {
      sign_alternations++;
    }
  }

  GetOutput() = sign_alternations;
  return true;
}

bool NikitinATestTaskSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace nikitin_a_vec_sign_rotation
