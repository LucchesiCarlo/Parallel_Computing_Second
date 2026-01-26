cmake_minimum_required(VERSION 4.1)

project(Second_Assignment LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_CXX_STANDARD_REQUIRED True)


if(UNIX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
else ()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Xpreprocessor -fopenmp -lomp")
endif ()



# You need to have installed OpenCV on the local machine
# Linux: sudo apt install libopencv-dev
find_package( OpenCV REQUIRED )
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit REQUIRED)

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

If (OPENMP_FOUND)
set (CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(ParallelKernel sequential_main.cpp)
target_link_libraries(ParallelKernel ${OpenCV_LIBS} )

add_executable(ParallelImg parallel_main.cpp)
target_link_libraries(ParallelImg ${OpenCV_LIBS} )

add_executable(Cuda cuda_main.cu src/cuda_kernel.cu)
target_link_libraries(Cuda ${OpenCV_LIBS} OpenMP::OpenMP_CXX)


add_executable(CudaTile cuda_main.cu src/cuda_kernel_tiling.cu)
target_link_libraries(CudaTile ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

add_executable(CudaBatch cuda_main_batch.cu src/cuda_kernel_batch.cu)
target_link_libraries(CudaBatch ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

### Test Sources ###
add_executable(SequentialTest test_sequential_main.cpp)
target_link_libraries(SequentialTest ${OpenCV_LIBS} )

add_executable(ParallelTest test_parallel_main.cpp)
target_link_libraries(ParallelTest ${OpenCV_LIBS} )


# to optimize for the specific GPU architecture
#set_property(TARGET Second_Assignment PROPERTY CUDA_ARCHITECTURES native)