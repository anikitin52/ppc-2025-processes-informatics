#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace nikitin_a_fox_algorithm {

using InType = int;
using OutType = int;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace nikitin_a_fox_algorithm
