project(gemm)

find_package(CUDA REQUIRED)

file (GLOB SOURCE_FILES "*.cu" )

cuda_add_library(mygemm SHARED ${SOURCE_FILES})

set_property(
    TARGET mygemm
    PROPERTY CUDA_ARCHITECTURES
    75 # V100
)


set(LIBRARY_OUTPUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../build")
