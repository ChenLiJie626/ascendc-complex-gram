cmake_minimum_required(VERSION 3.16)
project(ascendc_complex_gram_msopgen_skeleton LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(COMPLEX_GRAM_BUILD_HOST_TILING "Build standalone host tiling calculator" ON)

if(COMPLEX_GRAM_BUILD_HOST_TILING)
  add_executable(complex_gram_tiling op_host/complex_gram_tiling.cpp)
  target_compile_definitions(complex_gram_tiling PRIVATE COMPLEX_GRAM_TILING_STANDALONE)
endif()
