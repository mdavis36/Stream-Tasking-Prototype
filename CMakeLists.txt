cmake_minimum_required(VERSION 3.14)
project(STP LANGUAGES CXX CUDA)

set(EXAMPLE 0)
#add_definitions(-DEXAMPLE "${EXAMPLE}")
set(CMAKE_CUDA_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11 --default-stream per-thread -DEXAMPLE=${EXAMPLE}")



add_subdirectory(src)

add_executable(stp-exec src/main.cu)
