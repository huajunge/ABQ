# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# 最低CMake版本要求
cmake_minimum_required(VERSION 3.10)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置编译选项
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -fopenmp -O3 -DNDEBUG")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw -fopenmp -O3 -DNDEBUG")

# 手动查找HDF5库（不依赖find_package）
find_path(HDF5_INCLUDE_DIRS hdf5.h HINTS /usr/include /usr/local/include)
find_library(HDF5_CPP_LIBRARIES hdf5_cpp HINTS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)
find_library(HDF5_LIBRARIES hdf5 HINTS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)
find_library(HDF5_HL_LIBRARIES hdf5_hl HINTS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)
find_library(HDF5_CPP_HL_LIBRARIES hdf5_hl_cpp HINTS /usr/lib64 /usr/lib /usr/local/lib64 /usr/local/lib)

# 查找OpenMP
find_package(OpenMP REQUIRED COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# 设置Faiss库路径 - 使用构建目录中的库文件
set(FAISS_BUILD_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../faiss/libs)
set(FAISS_AVX512_LIB ${FAISS_BUILD_DIR}/libfaiss_avx512.a)
set(FAISS_BASE_LIB ${FAISS_BUILD_DIR}/libfaiss.a)

# 检查Faiss库文件是否存在
if(NOT EXISTS ${FAISS_AVX512_LIB})
    message(FATAL_ERROR "Faiss AVX512 library not found: ${FAISS_AVX512_LIB}. Please run 'make install' in the Faiss build directory first.")
endif()

if(NOT EXISTS ${FAISS_BASE_LIB})
    message(FATAL_ERROR "Faiss base library not found: ${FAISS_BASE_LIB}. Please run 'make install' in the Faiss build directory first.")
endif()

# 添加可执行文件
add_executable(nprobe_benchmark nprobe_benchmark.cpp)

# 链接库文件 - 直接链接静态库，注意链接顺序很重要
target_link_libraries(nprobe_benchmark 
    PRIVATE 
    # System libraries first
    pthread 
    dl 
    m 
    z
    # BLAS/LAPACK libraries (Faiss depends on these)
    openblas
    lapack
    # HDF5 libraries - include C++ libraries
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    # OpenMP library
    omp
    # Faiss libraries last (they depend on the above)
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)

# 包含目录
target_include_directories(nprobe_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    ${FAISS_INSTALL_DIR}/include
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)

# 设置编译定义
target_compile_definitions(nprobe_benchmark PRIVATE 
    -DFAISS_ENABLE_GPU=0
    -DHAVE_HDF5=1
)

# 设置RPATH，便于运行时找到依赖库
set_target_properties(nprobe_benchmark PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "/usr/local/lib64:/usr/lib64"
)

# 创建构建目标，可以直接构建而不需要外部CMakeLists
set_target_properties(nprobe_benchmark PROPERTIES
    EXCLUDE_FROM_ALL FALSE
)

# 添加自定义构建命令，确保库文件存在
add_custom_command(TARGET nprobe_benchmark PRE_BUILD
    COMMAND echo "Checking Faiss libraries..."
    COMMAND test -f ${FAISS_AVX512_LIB} && echo "Found libfaiss_avx512.a" || (echo "ERROR: libfaiss_avx512.a not found" && exit 1)
    COMMAND test -f ${FAISS_BASE_LIB} && echo "Found libfaiss.a" && ls -la ${FAISS_AVX512_LIB} || (echo "ERROR: libfaiss.a not found" && exit 1)
)

message(STATUS "Configured nprobe_benchmark for standalone build")
message(STATUS "Faiss AVX512 library: ${FAISS_AVX512_LIB}")
message(STATUS "Faiss base library: ${FAISS_BASE_LIB}")
message(STATUS "HDF5 includes: ${HDF5_INCLUDE_DIRS}")
message(STATUS "HDF5 libraries: ${HDF5_LIBRARIES}")

# Additional executables from demos
# =============================================================================

add_executable(nlist_abq EXCLUDE_FROM_ALL nlist_abq.cpp)
target_link_libraries(nlist_abq 
    PRIVATE 
    pthread 
    dl 
    m 
    z
    openblas
    lapack
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    omp
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)
target_include_directories(nlist_abq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)

add_executable(z_abq EXCLUDE_FROM_ALL z_abq.cpp)
target_link_libraries(z_abq 
    PRIVATE 
    pthread 
    dl 
    m 
    z
    openblas
    lapack
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    omp
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)
target_include_directories(z_abq PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)

add_executable(training_data EXCLUDE_FROM_ALL training_data.cpp)
target_link_libraries(training_data 
    PRIVATE 
    pthread 
    dl 
    m 
    z
    openblas
    lapack
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    omp
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)
target_include_directories(training_data PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)

add_executable(tail_latency EXCLUDE_FROM_ALL tail_latency.cpp)
target_link_libraries(tail_latency 
    PRIVATE 
    pthread 
    dl 
    m 
    z
    openblas
    lapack
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    omp
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)
target_include_directories(tail_latency PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)

add_executable(scalibility EXCLUDE_FROM_ALL scalibility.cpp)
target_link_libraries(scalibility 
    PRIVATE 
    pthread 
    dl 
    m 
    z
    openblas
    lapack
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    omp
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)
target_include_directories(scalibility PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)

add_executable(scalability_dataset EXCLUDE_FROM_ALL scalability_dataset.cpp)
target_link_libraries(scalability_dataset 
    PRIVATE 
    pthread 
    dl 
    m 
    z
    openblas
    lapack
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    omp
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)
target_include_directories(scalability_dataset PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)

add_executable(nprobe EXCLUDE_FROM_ALL nprobe.cpp)
target_link_libraries(nprobe 
    PRIVATE 
    pthread 
    dl 
    m 
    z
    openblas
    lapack
    ${HDF5_CPP_LIBRARIES}
    ${HDF5_LIBRARIES} 
    ${HDF5_HL_LIBRARIES}
    ${HDF5_CPP_HL_LIBRARIES}
    omp
    ${FAISS_AVX512_LIB}
    ${FAISS_BASE_LIB}
)
target_include_directories(nprobe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${HDF5_INCLUDE_DIRS}
    /usr/include
    /usr/local/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
)