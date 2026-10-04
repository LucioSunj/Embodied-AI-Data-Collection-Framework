# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
cmake_minimum_required(VERSION 3.14)
project(torch_isolation)

add_subdirectory("pinocchio_isolation/")

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

# Find Torch
find_package(Python3 REQUIRED COMPONENTS Interpreter)
if(DEFINED ENV{CONDA_PREFIX})
    set(_torch_cmake_root "$ENV{CONDA_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/torch/share/cmake")
    set(_torch_lib_dir "$ENV{CONDA_PREFIX}/lib/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages/torch/lib")
    set(Torch_DIR "${_torch_cmake_root}/Torch")
    set(Caffe2_DIR "${_torch_cmake_root}/Caffe2")
endif()
find_package(Torch REQUIRED)

# Find Eigen
find_package(Eigen3 REQUIRED)

include_directories(
    "include"
    "${TORCH_INCLUDE_DIRS}"
    "${EIGEN3_INCLUDE_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}/pinocchio_isolation/include"
)

add_library(torch_server_ops SHARED src/torch_server_ops.cpp)
set_target_properties(torch_server_ops PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torch_server_ops PRIVATE
    "${TORCH_LIBRARIES}"
)
if(DEFINED _torch_lib_dir AND DEFINED ENV{CONDA_PREFIX})
    set_target_properties(torch_server_ops PROPERTIES
        BUILD_RPATH "\$ORIGIN;${_torch_lib_dir};$ENV{CONDA_PREFIX}/lib"
        INSTALL_RPATH "\$ORIGIN;${_torch_lib_dir};$ENV{CONDA_PREFIX}/lib"
    )
endif()

add_library(torchrot SHARED src/rotations.cpp)
set_target_properties(torchrot PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torchrot PRIVATE
    "${TORCH_LIBRARIES}"
)
if(DEFINED _torch_lib_dir AND DEFINED ENV{CONDA_PREFIX})
    set_target_properties(torchrot PROPERTIES
        BUILD_RPATH "\$ORIGIN;${_torch_lib_dir};$ENV{CONDA_PREFIX}/lib"
        INSTALL_RPATH "\$ORIGIN;${_torch_lib_dir};$ENV{CONDA_PREFIX}/lib"
    )
endif()

add_library(torchscript_pinocchio SHARED src/pinocchio.cpp)
set_target_properties(torchscript_pinocchio PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_link_libraries(torchscript_pinocchio PRIVATE
    pinocchio_wrapper
    "${TORCH_LIBRARIES}"
    torchrot
)
if(DEFINED _torch_lib_dir AND DEFINED ENV{CONDA_PREFIX})
    set_target_properties(torchscript_pinocchio PROPERTIES
        BUILD_RPATH "\$ORIGIN;\$ORIGIN/pinocchio_isolation;${_torch_lib_dir};$ENV{CONDA_PREFIX}/lib"
        INSTALL_RPATH "\$ORIGIN;${_torch_lib_dir};$ENV{CONDA_PREFIX}/lib"
    )
endif()
