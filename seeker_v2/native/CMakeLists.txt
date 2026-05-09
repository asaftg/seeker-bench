# CMake for seeker_v2 native extensions (Phase 2.2+).
#
# Build (on Jetson, inside seeker_v2/native/):
#   python3 -m pip install --user pybind11
#   mkdir -p build && cd build
#   cmake -DCMAKE_BUILD_TYPE=Release ..
#   make -j$(nproc)
#
# Outputs (placed in seeker_v2/native/, alongside the source):
#   seeker_native.cpython-*-aarch64-linux-gnu.so   (Phase 2.2 — V4L2 capture)
#   seeker_nvjpeg.cpython-*-aarch64-linux-gnu.so   (Phase 2.3 — nvjpeg encode)
#
# Both modules are imported as `seeker_v2.native.<name>` from Python; the
# capture / thermal processes try the native extensions first and fall back
# to pure-Python paths if a module fails to load (e.g. on x86 / Windows
# during dev where the headers don't exist).

cmake_minimum_required(VERSION 3.16)
project(seeker_native CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Aggressive optimization on Xavier (Volta sm_72, ARMv8.2-A + crypto + fp16).
# -O3 + LTO gives ~10% on the AGC inner loop in microbench.
add_compile_options(-O3 -ffast-math -fno-plt)
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err)
    if(_ipo_ok)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endif()

# pybind11 — installed via pip (`python3 -m pip install pybind11`) or apt.
find_package(pybind11 REQUIRED)

# ──────────────────────────────────────────────────────────────────────
# Phase 2.2 — V4L2 mmap capture + AGC + RAW12 reinterpret
# ──────────────────────────────────────────────────────────────────────
pybind11_add_module(seeker_native MODULE v4l2_capture.cpp)
target_link_libraries(seeker_native PRIVATE pthread)
target_compile_definitions(seeker_native PRIVATE
    PYBIND11_DETAILED_ERROR_MESSAGES=1
)
# Drop the .so right next to the source so Python can import it as
# seeker_v2.native.seeker_native without a separate install step.
set_target_properties(seeker_native PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
)

# ──────────────────────────────────────────────────────────────────────
# Phase 2.3 — nvjpeg hardware JPEG encoder (Jetson only)
# ──────────────────────────────────────────────────────────────────────
# nvjpeg headers ship with JetPack at /usr/src/jetson_multimedia_api/.
# On non-Jetson hosts (x86 dev), skip silently — the Python side falls
# back to cv2.imencode.
set(JETSON_MM_API "/usr/src/jetson_multimedia_api")
if(EXISTS "${JETSON_MM_API}/include/NvJpegEncoder.h")
    message(STATUS "seeker_nvjpeg: building (found ${JETSON_MM_API})")

    # The multimedia samples don't ship a CMake module; pull the headers
    # and pre-built .so files directly.
    set(JETSON_MM_INCLUDES
        "${JETSON_MM_API}/include"
        "${JETSON_MM_API}/include/libjpeg-8b"
    )
    set(JETSON_MM_LIB_DIR "/usr/lib/aarch64-linux-gnu/tegra")

    pybind11_add_module(seeker_nvjpeg MODULE nvjpeg_encoder.cpp)
    target_include_directories(seeker_nvjpeg PRIVATE ${JETSON_MM_INCLUDES})
    target_link_directories(seeker_nvjpeg PRIVATE ${JETSON_MM_LIB_DIR})
    target_link_libraries(seeker_nvjpeg PRIVATE
        nvjpeg
        nvbufsurface
        nvbufsurftransform
        pthread
    )
    set_target_properties(seeker_nvjpeg PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
else()
    message(STATUS "seeker_nvjpeg: skipped (Jetson multimedia API not found at ${JETSON_MM_API} — non-Jetson host)")
endif()

# ──────────────────────────────────────────────────────────────────────
# Phase 2.4 — DLA TensorRT runner (built only when TRT is present)
# ──────────────────────────────────────────────────────────────────────
# Optional. The Python side (processes/inference.py) currently uses
# ultralytics + TRT engines through Python. The C++ runner here is
# wired in only if/when the iGPU contention from running both EO and
# thermal YOLO simultaneously becomes the bottleneck.
find_path(TRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/aarch64-linux-gnu /usr/include
)
find_library(TRT_NVINFER_LIB nvinfer
    PATHS /usr/lib/aarch64-linux-gnu
)
if(TRT_INCLUDE_DIR AND TRT_NVINFER_LIB AND EXISTS "${CMAKE_SOURCE_DIR}/dla_runner.cpp")
    message(STATUS "seeker_dla: building (TensorRT at ${TRT_INCLUDE_DIR})")
    pybind11_add_module(seeker_dla MODULE dla_runner.cpp)
    target_include_directories(seeker_dla PRIVATE ${TRT_INCLUDE_DIR})
    target_link_libraries(seeker_dla PRIVATE
        ${TRT_NVINFER_LIB}
        cudart
        pthread
    )
    set_target_properties(seeker_dla PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
else()
    message(STATUS "seeker_dla: skipped (TensorRT or dla_runner.cpp not found)")
endif()

message(STATUS "seeker_native build configured: type=${CMAKE_BUILD_TYPE}")
