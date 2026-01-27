//
// Created by giacomo on 1/20/26.
//

#ifndef SECOND_ASSIGNMENT_CUDA_KERNEL_CUH
#define SECOND_ASSIGNMENT_CUDA_KERNEL_CUH


#define MAX_K 7


__device__ inline unsigned char cudaClamping(float x) {
    if (x < 0.0f) x = 0.0f;
    if (x > 255.0f) x = 255.0f;
    x = std::roundf(x);
    return static_cast<unsigned char> (x);
}

__host__ cudaError_t loadKernel(float* kernel, int K);
__global__  void applyCudaKernel(unsigned char* in, unsigned char* out, int K, int W, int H, int C);





#endif //SECOND_ASSIGNMENT_CUDA_KERNEL_CUH

