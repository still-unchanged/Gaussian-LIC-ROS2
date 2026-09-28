cmake_minimum_required(VERSION 3.18)
project(gaussian_lic LANGUAGES CXX CUDA)

set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g -Wno-sign-compare -Wno-unused -Wno-comment -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -msse4.2")

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_eigen REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(pcl_conversions REQUIRED)

# These cache variables can be supplied with -D... at configure time.  When a
# matching environment variable is present, use it only as the initial value.
if(NOT DEFINED OpenCV_DIR AND DEFINED ENV{OpenCV_DIR} AND NOT "$ENV{OpenCV_DIR}" STREQUAL "")
  set(OpenCV_DIR "$ENV{OpenCV_DIR}" CACHE PATH "OpenCV CMake directory")
endif()
find_package(OpenCV 4 REQUIRED)

if(NOT DEFINED Torch_DIR AND DEFINED ENV{Torch_DIR} AND NOT "$ENV{Torch_DIR}" STREQUAL "")
  set(Torch_DIR "$ENV{Torch_DIR}" CACHE PATH "LibTorch CMake directory")
endif()
find_package(Torch REQUIRED)

if(NOT DEFINED TENSORRT_ROOT)
  if(DEFINED ENV{TENSORRT_ROOT} AND NOT "$ENV{TENSORRT_ROOT}" STREQUAL "")
    set(TENSORRT_ROOT "$ENV{TENSORRT_ROOT}" CACHE PATH "TensorRT installation root")
  else()
    set(TENSORRT_ROOT "" CACHE PATH "TensorRT installation root")
  endif()
endif()

find_path(TENSORRT_INCLUDE_DIR
  NAMES NvInfer.h
  HINTS "${TENSORRT_ROOT}/include" "/usr/include/x86_64-linux-gnu"
  REQUIRED
)
find_library(TENSORRT_NVINFER_LIBRARY
  NAMES nvinfer
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" "/usr/lib/x86_64-linux-gnu"
  REQUIRED
)
find_library(TENSORRT_NVONNXPARSER_LIBRARY
  NAMES nvonnxparser
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" "/usr/lib/x86_64-linux-gnu"
  REQUIRED
)
find_library(TENSORRT_NVINFER_PLUGIN_LIBRARY
  NAMES nvinfer_plugin
  HINTS "${TENSORRT_ROOT}/lib" "${TENSORRT_ROOT}/lib64" "/usr/lib/x86_64-linux-gnu"
  REQUIRED
)

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CUDAToolkit_ROOT} AND NOT "$ENV{CUDAToolkit_ROOT}" STREQUAL "")
  set(CUDAToolkit_ROOT "$ENV{CUDAToolkit_ROOT}" CACHE PATH "CUDA toolkit root")
endif()
find_package(CUDAToolkit REQUIRED)
find_library(FFI_LIB ffi REQUIRED)
# PCL/VTK pull in system GDAL/NetCDF, which require the system OpenSSL curl ABI.
# LibTorch adds a Conda rpath-link, so make the compatible system curl explicit.
find_library(SYSTEM_CURL_LIBRARY NAMES curl
  PATHS /usr/lib/x86_64-linux-gnu /lib/x86_64-linux-gnu
  NO_DEFAULT_PATH REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(YAML_CPP REQUIRED yaml-cpp>=0.5)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)

set(ROS2_DEPENDENCIES
  rclcpp
  ament_index_cpp
  std_msgs
  nav_msgs
  sensor_msgs
  geometry_msgs
  tf2
  tf2_ros
  tf2_eigen
  cv_bridge
  pcl_conversions
)

add_executable(gs_mapping
  src/mapping.cpp
  src/gaussian.cpp
  src/tinyply.cpp
  src/depth_completer.cpp
  src/simple-knn/spatial.cu
  src/simple-knn/simple_knn.cu
  src/fused-ssim/ssim.cu
  src/rasterizer/renderer.cpp
  src/rasterizer/rasterizer.cpp
  src/rasterizer/rasterize_points.cu
  src/rasterizer/cuda_rasterizer/adam.cu
  src/rasterizer/cuda_rasterizer/forward.cu
  src/rasterizer/cuda_rasterizer/backward.cu
  src/rasterizer/cuda_rasterizer/rasterizer_impl.cu
)

target_include_directories(gs_mapping PRIVATE
  src
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIRS}
  ${Python3_INCLUDE_DIRS}
  ${TENSORRT_INCLUDE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

ament_target_dependencies(gs_mapping ${ROS2_DEPENDENCIES})

target_link_libraries(gs_mapping
  ${OpenCV_LIBS}
  ${TORCH_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
  ${Python3_LIBRARIES}
  ${FFI_LIB}
  ${SYSTEM_CURL_LIBRARY}
  CUDA::cudart
  ${TENSORRT_NVINFER_LIBRARY}
  ${TENSORRT_NVONNXPARSER_LIBRARY}
  ${TENSORRT_NVINFER_PLUGIN_LIBRARY}
)

install(TARGETS gs_mapping
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY config ckpt launch src/lpips
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS
  scripts/m3dgr_adapter.py
  scripts/fuse_tsdf_mesh.py
  tools/inspect_m3dgr_avia_projection.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
