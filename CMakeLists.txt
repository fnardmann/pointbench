cmake_minimum_required(VERSION 3.18)
project(pointbench)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${CMAKE_SOURCE_DIR})

add_library(sq_distance_cpu STATIC cpu/sq_distance.cpp)

find_package(CUDA)
if(CUDA_FOUND)
    cuda_add_library(sq_distance_cuda STATIC cuda/sq_distance.cu)
    target_compile_options(sq_distance_cuda PRIVATE --use_fast_math)
endif()