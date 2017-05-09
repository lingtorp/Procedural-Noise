cmake_minimum_required(VERSION 3.6)
project(Noise)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp vector.hpp noise.hpp)
add_executable(Noise ${SOURCE_FILES})

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
target_link_libraries(Noise ${SDL2_LIBRARIES})
