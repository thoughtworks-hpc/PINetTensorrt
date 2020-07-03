cmake_minimum_required(VERSION 3.10)

project(PINetTensorrt)

# add_compile_options("-g")
add_compile_options("-o2")

find_package(OpenCV REQUIRED)

set(TEGRA_LIB_DIR /usr/lib/aarch64-linux-gnu/tegra)
set(CUDA_INSTALL_DIR /usr/local/cuda)
set(CUDA_INCLUDE_DIR ${CUDA_INSTALL_DIR}/include)
set(CUDA_LIB_DIR ${CUDA_INSTALL_DIR}/lib64)

include_directories(common ${CUDA_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

aux_source_directory(. SRCS)
aux_source_directory(common COMMON_SRCS)

link_directories(${CUDA_LIB_DIR} ${TEGRA_LIB_DIR} )
add_executable(${PROJECT_NAME} ${COMMON_SRCS} ${SRCS})

set(CUDA_LIB cuda cudnn cublas cudart culibos)
set(NV_LIB nvinfer nvparsers nvinfer_plugin nvonnxparser)

target_link_libraries(${PROJECT_NAME} ${CUDA_LIB} ${NV_LIB} ${OpenCV_LIBS})