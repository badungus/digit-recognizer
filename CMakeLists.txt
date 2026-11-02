cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

project(digit-recognizer)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(FetchContent)

FetchContent_Declare(
    raylib
    GIT_REPOSITORY https://github.com/raysan5/raylib.git
    GIT_TAG 5.5
    GIT_SHALLOW TRUE
)

FetchContent_MakeAvailable(raylib)


add_executable(digit-recognizer
main.cpp
nn.cpp
visualizer.cpp
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

target_link_libraries(digit-recognizer PRIVATE raylib)

target_compile_definitions(digit-recognizer PRIVATE
  DATA_DIR="${CMAKE_SOURCE_DIR}/archive")

target_compile_definitions(digit-recognizer PRIVATE
  WEIGHTS_DIR="${CMAKE_SOURCE_DIR}/weights")
