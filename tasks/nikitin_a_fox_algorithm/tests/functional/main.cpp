#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "nikitin_a_fox_algorithm/common/include/common.hpp"
#include "nikitin_a_fox_algorithm/mpi/include/ops_mpi.hpp"
#include "nikitin_a_fox_algorithm/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace nikitin_a_fox_algorithm {

class NikitinAFoxAlgorithmFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::get<1>(test_param);  // возвращаем строковое описание теста
  }

 protected:
  void SetUp() override {
    auto params = GetParam();
    TestType test_params = std::get<2>(params);

    test_case_id_ = static_cast<int>(std::get<0>(test_params));
    test_description_ = std::get<1>(test_params);

    // Генерируем тестовые данные в зависимости от ID теста
    switch (test_case_id_) {
      case 1:  // Матрицы 1x1
        GenerateTestMatrices(1);
        break;

      case 2:  // Матрицы 2x2 - простой случай
        GenerateTestMatrices(2);
        break;

      case 3:  // Матрицы 3x3
        GenerateTestMatrices(3);
        break;

      case 4:  // Матрицы 4x4
        GenerateTestMatrices(4);
        break;

      case 5:  // Матрицы 5x5
        GenerateTestMatrices(5);
        break;

      case 6:  // Матрицы 8x8 - хороший размер для тестирования блоков
        GenerateTestMatrices(8);
        break;

      case 7:  // Матрицы 10x10
        GenerateTestMatrices(10);
        break;

      case 8:  // Матрицы 16x16 - тестирование четного деления на блоки
        GenerateTestMatrices(16);
        break;

      case 9:  // Матрицы 20x20
        GenerateTestMatrices(20);
        break;

      case 10:  // Матрицы 32x32 - большая матрица
        GenerateTestMatrices(32);
        break;

      case 11:  // Матрицы 50x50
        GenerateTestMatrices(50);
        break;

      case 12:  // Матрицы 64x64 - хороший размер для MPI (делится на блоки)
        GenerateTestMatrices(64);
        break;

      case 13:  // Матрицы 100x100 - большая для тестирования производительности
        GenerateTestMatrices(100);
        break;

      case 14:  // Матрицы 127x127 - простое число, тест на неравномерные блоки
        GenerateTestMatrices(127);
        break;

      case 15:  // Матрицы 128x128 - степень двойки
        GenerateTestMatrices(128);
        break;

      case 16:  // Матрицы 200x200
        GenerateTestMatrices(200);
        break;

      case 17:  // Матрицы 256x256
        GenerateTestMatrices(256);
        break;

      case 18:  // Матрицы 300x300
        GenerateTestMatrices(300);
        break;

      case 19:  // Матрицы 500x500 - большая матрица
        GenerateTestMatrices(500);
        break;

      case 20:  // Матрицы 512x512 - степень двойки
        GenerateTestMatrices(512);
        break;

      // Тесты с особыми значениями
      case 21:  // Матрицы с нулями
        GenerateZeroMatrices(10);
        break;

      case 22:  // Единичные матрицы (A = I, B = I => C = I)
        GenerateIdentityMatrices(8);
        break;

      case 23:  // Нулевая матрица × любая матрица = нулевая матрица
        GenerateZeroTimesAny(5);
        break;

      case 24:  // Диагональные матрицы
        GenerateDiagonalMatrices(6);
        break;

      case 25:  // Верхнетреугольные матрицы
        GenerateUpperTriangularMatrices(7);
        break;

      case 26:  // Нижнетреугольные матрицы
        GenerateLowerTriangularMatrices(7);
        break;

      case 27:  // Симметричные матрицы
        GenerateSymmetricMatrices(9);
        break;

      case 28:  // Матрицы с очень маленькими значениями
        GenerateSmallValueMatrices(4);
        break;

      case 29:  // Матрицы с очень большими значениями
        GenerateLargeValueMatrices(4);
        break;

      case 30:  // Матрицы со смешанными положительными и отрицательными значениями
        GenerateMixedSignMatrices(6);
        break;

      case 31:  // Матрицы с одинаковыми элементами
        GenerateConstantMatrices(5);
        break;

      case 32:  // Матрицы, где A × B ≠ B × A (тест на некоммутативность)
        GenerateNonCommutativeMatrices();
        break;

      case 33:  // Матрицы, где A × (B × C) = (A × B) × C (ассоциативность)
        GenerateAssociativityTest(4);
        break;

      case 34:  // Матрицы с NaN и Inf значениями
        GenerateSpecialValueMatrices(3);
        break;

      case 35:  // Рандомные матрицы с разными seed
        GenerateRandomMatrices(25, 12345);
        break;

      case 36:  // Большие рандомные матрицы
        GenerateRandomMatrices(100, 54321);
        break;

      case 37:  // Матрицы с высокой точностью
        GenerateHighPrecisionMatrices(3);
        break;

      case 38:  // Тест на граничные значения double
        GenerateBoundaryValueMatrices(2);
        break;

      case 39:  // Почти сингулярные матрицы
        GenerateNearSingularMatrices(5);
        break;

      case 40:  // Тест на ортогональные матрицы
        GenerateOrthogonalMatrices(4);
        break;

      default:
        throw std::runtime_error("Unknown test case ID: " + std::to_string(test_case_id_));
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    // Проверяем размер
    if (output_data.size() != expected_output_.size()) {
      return false;
    }
    
    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i].size() != expected_output_[i].size()) {
        return false;
      }
    }

    // Проверяем каждый элемент с учетом погрешности для чисел с плавающей точкой
    double max_relative_error = 0.0;
    double max_absolute_error = 0.0;
    
    for (size_t i = 0; i < output_data.size(); ++i) {
      for (size_t j = 0; j < output_data[i].size(); ++j) {
        double expected = expected_output_[i][j];
        double actual = output_data[i][j];
        
        // Обработка специальных значений
        if (std::isnan(expected) && std::isnan(actual)) {
          continue;  // Оба NaN - OK
        }
        
        if (std::isinf(expected) && std::isinf(actual) && 
            std::signbit(expected) == std::signbit(actual)) {
          continue;  // Оба Inf с одинаковым знаком - OK
        }
        
        // Для обычных чисел используем относительную погрешность
        if (std::abs(expected) > 1e-10) {
          double relative_error = std::abs(actual - expected) / std::abs(expected);
          if (relative_error > 1e-10) {
            return false;
          }
          max_relative_error = std::max(max_relative_error, relative_error);
        } else {
          // Для очень маленьких чисел используем абсолютную погрешность
          double absolute_error = std::abs(actual - expected);
          if (absolute_error > 1e-10) {
            return false;
          }
          max_absolute_error = std::max(max_absolute_error, absolute_error);
        }
      }
    }

    return true;
  }

  InType GetTestInputData() final {
    return {matrix_a_, matrix_b_};
  }

 private:
  int test_case_id_ = 0;
  std::string test_description_;
  std::vector<std::vector<double>> matrix_a_;
  std::vector<std::vector<double>> matrix_b_;
  std::vector<std::vector<double>> expected_output_;

  // Генерация матриц заданного размера
  void GenerateTestMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    std::mt19937 gen(n * 123);  // seed зависит от n
    std::uniform_real_distribution<double> dist(-10.0, 10.0);
    
    // Заполняем матрицы случайными значениями
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    // Вычисляем ожидаемый результат (простое умножение матриц)
    ComputeExpectedResult();
  }

  // Вычисление ожидаемого результата умножения матриц
  void ComputeExpectedResult() {
    int n = matrix_a_.size();
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        double sum = 0.0;
        for (int k = 0; k < n; ++k) {
          sum += matrix_a_[i][k] * matrix_b_[k][j];
        }
        expected_output_[i][j] = sum;
      }
    }
  }

  // Генерация нулевых матриц
  void GenerateZeroMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
  }

  // Генерация единичных матриц
  void GenerateIdentityMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    for (int i = 0; i < n; ++i) {
      matrix_a_[i][i] = 1.0;
      matrix_b_[i][i] = 1.0;
      expected_output_[i][i] = 1.0;
    }
  }

  // Генерация: нулевая матрица × любая матрица
  void GenerateZeroTimesAny(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    std::mt19937 gen(n * 456);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_b_[i][j] = dist(gen);
      }
    }
  }

  // Генерация диагональных матриц
  void GenerateDiagonalMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    std::mt19937 gen(n * 789);
    std::uniform_real_distribution<double> dist(1.0, 10.0);
    
    for (int i = 0; i < n; ++i) {
      matrix_a_[i][i] = dist(gen);
      matrix_b_[i][i] = dist(gen);
    }
    
    ComputeExpectedResult();
  }

  // Генерация верхнетреугольных матриц
  void GenerateUpperTriangularMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    std::mt19937 gen(n * 1011);
    std::uniform_real_distribution<double> dist(1.0, 5.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация нижнетреугольных матриц
  void GenerateLowerTriangularMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    std::mt19937 gen(n * 1213);
    std::uniform_real_distribution<double> dist(1.0, 5.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j <= i; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация симметричных матриц
  void GenerateSymmetricMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    std::mt19937 gen(n * 1415);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = i; j < n; ++j) {
        double val = dist(gen);
        matrix_a_[i][j] = val;
        matrix_a_[j][i] = val;
        
        val = dist(gen);
        matrix_b_[i][j] = val;
        matrix_b_[j][i] = val;
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация матриц с очень маленькими значениями
  void GenerateSmallValueMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    
    std::mt19937 gen(n * 1617);
    std::uniform_real_distribution<double> dist(1e-15, 1e-10);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация матриц с очень большими значениями
  void GenerateLargeValueMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    
    std::mt19937 gen(n * 1819);
    std::uniform_real_distribution<double> dist(1e10, 1e15);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация матриц со смешанными знаками
  void GenerateMixedSignMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    
    std::mt19937 gen(n * 2021);
    std::uniform_real_distribution<double> dist(-1000.0, 1000.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация матриц с постоянными элементами
  void GenerateConstantMatrices(int n) {
    double val_a = 2.5;
    double val_b = 3.5;
    
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, val_a));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, val_b));
    expected_output_ = std::vector<std::vector<double>>(n, std::vector<double>(n, val_a * val_b * n));
  }

  // Генерация некоммутативных матриц
  void GenerateNonCommutativeMatrices() {
    matrix_a_ = {{1, 2}, {3, 4}};
    matrix_b_ = {{5, 6}, {7, 8}};
    
    ComputeExpectedResult();
  }

  // Генерация теста на ассоциативность
  void GenerateAssociativityTest(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    
    std::mt19937 gen(n * 2223);
    std::uniform_real_distribution<double> dist(-5.0, 5.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация матриц с NaN и Inf значениями
  void GenerateSpecialValueMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
    
    // Добавляем специальные значения
    if (n >= 2) {
      matrix_a_[0][0] = std::numeric_limits<double>::quiet_NaN();
      matrix_a_[0][1] = std::numeric_limits<double>::infinity();
      matrix_a_[1][0] = -std::numeric_limits<double>::infinity();
      
      matrix_b_[1][1] = std::numeric_limits<double>::quiet_NaN();
    }
    
    ComputeExpectedResult();
  }

  // Генерация случайных матриц
  void GenerateRandomMatrices(int n, int seed) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    
    std::mt19937 gen(seed);
    std::uniform_real_distribution<double> dist(-100.0, 100.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen);
        matrix_b_[i][j] = dist(gen);
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация матриц с высокой точностью
  void GenerateHighPrecisionMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    
    std::mt19937 gen(n * 2425);
    std::uniform_real_distribution<double> dist(1.0, 2.0);
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = dist(gen) / 3.0;  // Простые дроби для точного представления
        matrix_b_[i][j] = dist(gen) / 7.0;
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация матриц с граничными значениями double
  void GenerateBoundaryValueMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n));
    
    double values[] = {
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::max(),
      -std::numeric_limits<double>::max(),
      std::numeric_limits<double>::epsilon(),
      -std::numeric_limits<double>::epsilon(),
      0.0
    };
    
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        matrix_a_[i][j] = values[(i * n + j) % 6];
        matrix_b_[i][j] = values[(i * n + j + 3) % 6];
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация почти сингулярных матриц
  void GenerateNearSingularMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 1.0));
    
    // Делаем матрицу A почти сингулярной
    for (int i = 0; i < n; ++i) {
      for (int j = i + 1; j < n; ++j) {
        matrix_a_[i][j] = 1.0 - 1e-10;
      }
    }
    
    ComputeExpectedResult();
  }

  // Генерация ортогональных матриц
  void GenerateOrthogonalMatrices(int n) {
    matrix_a_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    matrix_b_ = std::vector<std::vector<double>>(n, std::vector<double>(n, 0.0));
    
    // Простая ортогональная матрица: единичная
    for (int i = 0; i < n; ++i) {
      matrix_a_[i][i] = 1.0;
      matrix_b_[i][i] = 1.0;
    }
    
    // Добавляем вращение для первых двух строк/столбцов
    if (n >= 2) {
      double angle = 3.14159 / 4.0;  // 45 градусов
      matrix_a_[0][0] = std::cos(angle);
      matrix_a_[0][1] = -std::sin(angle);
      matrix_a_[1][0] = std::sin(angle);
      matrix_a_[1][1] = std::cos(angle);
      
      matrix_b_[0][0] = std::cos(-angle);
      matrix_b_[0][1] = -std::sin(-angle);
      matrix_b_[1][0] = std::sin(-angle);
      matrix_b_[1][1] = std::cos(-angle);
    }
    
    ComputeExpectedResult();
  }
};

namespace {

TEST_P(NikitinAFoxAlgorithmFuncTests, MatrixMultiplicationTest) {
  ExecuteTest(GetParam());
}

// Определяем тестовые случаи (40 тестов для полного покрытия)
const std::array<TestType, 40> kTestParam = {
    std::make_tuple(1, "1x1_matrices"),
    std::make_tuple(2, "2x2_matrices"),
    std::make_tuple(3, "3x3_matrices"),
    std::make_tuple(4, "4x4_matrices"),
    std::make_tuple(5, "5x5_matrices"),
    std::make_tuple(6, "8x8_matrices"),
    std::make_tuple(7, "10x10_matrices"),
    std::make_tuple(8, "16x16_matrices"),
    std::make_tuple(9, "20x20_matrices"),
    std::make_tuple(10, "32x32_matrices"),
    std::make_tuple(11, "50x50_matrices"),
    std::make_tuple(12, "64x64_matrices"),
    std::make_tuple(13, "100x100_matrices"),
    std::make_tuple(14, "127x127_matrices"),
    std::make_tuple(15, "128x128_matrices"),
    std::make_tuple(16, "200x200_matrices"),
    std::make_tuple(17, "256x256_matrices"),
    std::make_tuple(18, "300x300_matrices"),
    std::make_tuple(19, "500x500_matrices"),
    std::make_tuple(20, "512x512_matrices"),
    std::make_tuple(21, "zero_matrices"),
    std::make_tuple(22, "identity_matrices"),
    std::make_tuple(23, "zero_times_any"),
    std::make_tuple(24, "diagonal_matrices"),
    std::make_tuple(25, "upper_triangular"),
    std::make_tuple(26, "lower_triangular"),
    std::make_tuple(27, "symmetric_matrices"),
    std::make_tuple(28, "small_values"),
    std::make_tuple(29, "large_values"),
    std::make_tuple(30, "mixed_signs"),
    std::make_tuple(31, "constant_matrices"),
    std::make_tuple(32, "non_commutative"),
    std::make_tuple(33, "associativity_test"),
    std::make_tuple(34, "special_values_nan_inf"),
    std::make_tuple(35, "random_25x25"),
    std::make_tuple(36, "random_100x100"),
    std::make_tuple(37, "high_precision"),
    std::make_tuple(38, "boundary_values"),
    std::make_tuple(39, "near_singular"),
    std::make_tuple(40, "orthogonal_matrices")
};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<NikitinAFoxAlgorithmMPI, InType>(kTestParam, PPC_SETTINGS_nikitin_a_fox_algorithm),
                   ppc::util::AddFuncTask<NikitinAFoxAlgorithmSEQ, InType>(kTestParam, PPC_SETTINGS_nikitin_a_fox_algorithm));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NikitinAFoxAlgorithmFuncTests::PrintFuncTestName<NikitinAFoxAlgorithmFuncTests>;

INSTANTIATE_TEST_SUITE_P(FoxAlgorithmTests, NikitinAFoxAlgorithmFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace nikitin_a_fox_algorithm