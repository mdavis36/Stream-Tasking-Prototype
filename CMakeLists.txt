cmake_minimum_required(VERSION 3.15)
project(STP LANGUAGES CXX CUDA)

add_subdirectory(src)

add_executable(stp-exec src/main.cu)
