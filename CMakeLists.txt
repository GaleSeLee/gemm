cmake_minimum_required(VERSION 3.10)

project(gemm)

set(PROJECT_NAME gemm)

include_directories(${PROJECT_SOURCE_DIR}/include)
set(CMAKE_CXX_STANDARD 17)

SET(CMAKE_BUILD_TYPE Debug)
SET(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS} -O0 -Wall")

add_executable(${PROJECT_NAME}
               ${PROJECT_SOURCE_DIR}/src/main.cpp
               ${PROJECT_SOURCE_DIR}/src/gemm_baseline.cpp)
