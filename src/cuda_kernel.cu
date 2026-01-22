//
// Created by giacomo on 1/19/26.
//


#include "cuda_kernel.cuh"
__constant__ float cuda_kernel[MAX_K*MAX_K];
__host__ void loadKernel(float* kernel, int K) {

    cudaMemcpyToSymbol(cuda_kernel, kernel, sizeof(float)*K*K);
}

__global__  void applyCudaKernel(unsigned char* in, unsigned char* out, int K, int W, int H, int C) {
    int center = K / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) {return;}

    for (int c = 0; c < C; c++) {
        float result = 0;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                const int xIdx = j - center + x;
                const int yIdx = i - center + y;
                if (!(xIdx < 0 || xIdx >= W || yIdx < 0 || yIdx >= H)) {
                    const int inputIdx = yIdx * W * C + xIdx * C + c;
                    result += cuda_kernel[i * K + j] * in[inputIdx];
                }
            }
        }
        out[y * W * C + x * C + c] = cudaClamping(result);
    }

}