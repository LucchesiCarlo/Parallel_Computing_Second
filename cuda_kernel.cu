//
// Created by giacomo on 1/19/26.
//

#include "src/kernel_functions.h"

__device__ inline unsigned char cudaClamping(float x) {
    if (x < 0.0f) x = 0.0f;
    if (x > 255.0f) x = 255.0f;
    x = std::roundf(x);
    return static_cast<unsigned char> (x);
}


__global__  void applyCudaKernel(unsigned char* in, unsigned char* out, float* kernel, int K, int W, int H, int C) {
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
                    result += kernel[i * K + j] * in[inputIdx];
                }
            }
        }
        out[y * W * C + x * C + c] = cudaClamping(result);
    }

}