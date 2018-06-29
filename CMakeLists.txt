cmake_minimum_required(VERSION 3.6)
project(Noise)

set(CMAKE_CXX_STANDARD 11)

# Add pthread when compiling for Linux
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
if (UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread") # pthread for Linux
endif(UNIX)

set(SOURCE_FILES main.cpp vector.hpp noise.hpp)
add_executable(Noise ${SOURCE_FILES})

find_package(SDL2 REQUIRED)
include_directories(${SDL2_INCLUDE_DIRS})
target_link_libraries(Noise ${SDL2_LIBRARIES})

find_package(GLEW REQUIRED)
include_directories(${GLEW_INCLUDE_DIRS})
target_link_libraries(Noise ${GLEW_LIBRARIES})

find_package(OpenGL REQUIRED)
include_directories(${OPENGL_INCLUDE_DIRS})
target_link_libraries(Noise ${OPENGL_LIBRARIES})