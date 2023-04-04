cmake_minimum_required(VERSION 3.18)

project(gemm)

set(PROJECT_BIN_NAME gemm)

include_directories(./include)

include(CheckLanguage)
check_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
enable_language(CUDA)
set_property(
    TARGET ${ABACUS_BIN_NAME}
    PROPERTY CUDA_ARCHITECTURES
    75 # Turing
)

add_subdirectory(src)

add_executable(${PROJECT_BIN_NAME}
                src/main.cpp
                )

target_link_libraries(
    ${PROJECT_BIN_NAME}
    mygemm
)

include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
