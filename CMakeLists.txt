cmake_minimum_required(VERSION 3.15)
project(onnx_cpu_inference VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Find ONNX Runtime
# You can set ONNXRUNTIME_ROOT environment variable or pass -DONNXRUNTIME_ROOT=<path>
if(DEFINED ENV{ONNXRUNTIME_ROOT})
    set(ONNXRUNTIME_ROOT $ENV{ONNXRUNTIME_ROOT})
endif()

if(NOT DEFINED ONNXRUNTIME_ROOT)
    message(WARNING "ONNXRUNTIME_ROOT not set. Please download ONNX Runtime from https://github.com/microsoft/onnxruntime/releases")
    message(WARNING "Usage: cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..")
endif()

# Include directories
include_directories(${ONNXRUNTIME_ROOT}/include)

# Link directories
link_directories(${ONNXRUNTIME_ROOT}/lib)

# Main executable
add_executable(inference_pipeline main.cpp inference_pipeline.cpp)

# Link ONNX Runtime
target_link_libraries(inference PRIVATE onnxruntime)

# Add include directory for our headers
target_include_directories(inference PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Optional: Install target
install(TARGETS inference DESTINATION bin)
