# CMake for seeker_v2 native extensions (Phase 2.2+).
# Build with:
#   mkdir build && cd build && cmake .. && make -j4
# Output: seeker_native.cpython-*.so in seeker_v2/native/

cmake_minimum_required(VERSION 3.16)
project(seeker_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Find pybind11 (system-installed via pip or apt)
find_package(pybind11 REQUIRED)

# Phase 2.2: V4L2 capture extension
# pybind11_add_module(seeker_native v4l2_capture.cpp)
# target_link_libraries(seeker_native PRIVATE pthread)

message(STATUS "seeker_native: scaffolding only; v4l2_capture.cpp not yet implemented")
