
cmake_minimum_required(VERSION 3.13)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PICO_BOARD pico2 CACHE STRING "Board type")

include(${CMAKE_CURRENT_LIST_DIR}/../pico_sdk_import.cmake)

project(bearing_tinyml C CXX ASM)

pico_sdk_init()

# TensorFlow Lite Micro library
add_subdirectory(../lib/pico-tflmicro tflm_build)

# TinyML application
add_executable(bearing_tinyml
    src/main.cpp
)

target_include_directories(bearing_tinyml PRIVATE
    ${CMAKE_CURRENT_LIST_DIR}/model
)

target_link_libraries(bearing_tinyml
    pico_stdlib
    pico-tflmicro
)

pico_enable_stdio_usb(bearing_tinyml 1)
pico_enable_stdio_uart(bearing_tinyml 0)

pico_add_extra_outputs(bearing_tinyml)
