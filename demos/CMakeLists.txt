# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

find_package(OpenMP REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS CXX)
find_package(HDF5 REQUIRED COMPONENTS CXX)
#target_link_libraries(demo_block_ivf_indexing_file_sift PRIVATE faiss_avx2  ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX  ${HDF5_HL_LIBRARIES} dl m z)
add_executable(demo_flie_reader EXCLUDE_FROM_ALL demo_flie_reader.cpp)
target_link_libraries(demo_flie_reader PRIVATE faiss faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(demo_flie_reader PRIVATE ${OpenMP_CXX_FLAGS})

add_executable(nprobe EXCLUDE_FROM_ALL nprobe.cpp)
target_link_libraries(nprobe PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(nprobe PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)

add_executable(nlist_abq EXCLUDE_FROM_ALL nlist_abq.cpp)
target_link_libraries(nlist_abq PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(nlist_abq PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)

add_executable(z_abq EXCLUDE_FROM_ALL z_abq.cpp)
target_link_libraries(z_abq PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(z_abq PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)


add_executable(training_data EXCLUDE_FROM_ALL training_data.cpp)
target_link_libraries(training_data PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(training_data PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)

add_executable(tail_latency EXCLUDE_FROM_ALL tail_latency.cpp)
target_link_libraries(tail_latency PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(tail_latency PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)

add_executable(distribution EXCLUDE_FROM_ALL distribution.cpp)
target_link_libraries(distribution PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(distribution PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)

add_executable(distribution_new EXCLUDE_FROM_ALL distribution_new.cpp)
target_link_libraries(distribution_new PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(distribution_new PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)

add_executable(scalibility EXCLUDE_FROM_ALL scalibility.cpp)
target_link_libraries(scalibility PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(scalibility PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)

add_executable(scalability_dataset EXCLUDE_FROM_ALL scalability_dataset.cpp)
target_link_libraries(scalability_dataset PRIVATE faiss_avx512 ${HDF5_LIBRARIES} OpenMP::OpenMP_CXX ${HDF5_HL_LIBRARIES} dl m z)
target_compile_options(scalability_dataset PRIVATE ${OpenMP_CXX_FLAGS} -mavx512f -mavx512cd -mavx512vl -mavx512dq -mavx512bw)