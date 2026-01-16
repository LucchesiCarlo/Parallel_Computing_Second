cmake_minimum_required(VERSION 4.1)
project(Second_Assignment)

set(CMAKE_CXX_STANDARD 20)
# You need to have installed OpenCV on the local machine
# Linux: sudo apt install libopencv-dev
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )


if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options("-fsanitize=address,leak,undefined")
    add_link_options("-fsanitize=address,leak,undefined")
    add_compile_options("-g")
elseif (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options("-O3")
    add_compile_options("-mfma")
    add_compile_options("-march=native")
    add_compile_options("-fvectorize")
    add_compile_options("-fno-math-errno")
endif()

add_executable(Second_Assignment main.cpp sequential_kernel.cpp)
target_link_libraries(Second_Assignment ${OpenCV_LIBS} )
