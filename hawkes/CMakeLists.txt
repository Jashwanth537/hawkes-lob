cmake_minimum_required(VERSION 3.15)
project(hawkes CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
add_compile_options(-O3 -march=native)

include(FetchContent)
FetchContent_Declare(LBFGSpp
  GIT_REPOSITORY https://github.com/yixuan/LBFGSpp.git
  GIT_TAG master)
FetchContent_MakeAvailable(LBFGSpp)

find_package(Eigen3 REQUIRED)

add_executable(hawkes src/main.cpp src/hawkes.cpp)
target_include_directories(hawkes PRIVATE src)
target_link_libraries(hawkes PRIVATE lbfgspp Eigen3::Eigen)
