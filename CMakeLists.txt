cmake_minimum_required(VERSION 4.1)
project(Second_Assignment)

set(CMAKE_CXX_STANDARD 20)
# You need to have installed OpenCV on the local machine
# Linux: sudo apt install libopencv-dev
find_package( OpenCV REQUIRED )

include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable(Second_Assignment main.cpp sequential_kernel.cpp)
target_link_libraries(Second_Assignment ${OpenCV_LIBS} )
