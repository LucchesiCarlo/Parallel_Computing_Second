cmake_minimum_required(VERSION 4.1)
project(Second_Assignment)

set(CMAKE_CXX_STANDARD 20)


if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp")
endif ()

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
    add_compile_options("-fno-math-errno")
    add_compile_options("-ffast-math")
endif()

add_executable(ParallelKernel sequential_main.cpp)
target_link_libraries(ParallelKernel ${OpenCV_LIBS} )

add_executable(ParallelImg parallel_main.cpp)
target_link_libraries(ParallelImg ${OpenCV_LIBS} )

### Test Sources ###
add_executable(SequentialTest test_sequential_main.cpp)
target_link_libraries(SequentialTest ${OpenCV_LIBS} )

add_executable(ParallelTest test_parallel_main.cpp)
target_link_libraries(ParallelTest ${OpenCV_LIBS} )

