cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(digit-recognizer)

set(CMAKE_CXX_STANDARD 17)

add_executable(digit-recognizer
main.cpp
nn.cpp
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

target_compile_definitions(digit-recognizer PRIVATE
  DATA_DIR="${CMAKE_SOURCE_DIR}/archive")
