//
// Created by giacomo on 1/20/26.
//


#include "cuda_kernel.cuh"

__constant__ float cuda_kernel[MAX_K*MAX_K];
__host__ cudaError_t loadKernel(float* kernel, int K) {

    return cudaMemcpyToSymbol(cuda_kernel, kernel, sizeof(float)*K*K);
}


__global__  void applyCudaKernel(unsigned char* in, unsigned char* out, int K, int W, int H, int C) {

    int center = K / 2;
    int dimTile = (blockDim.x + (K-1))*(blockDim.y + (K-1));

    extern __shared__ unsigned char sMem[];


    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int tileOffset = dimTile/(blockDim.x*blockDim.y);

    for (int i=0; i<=tileOffset; i++) {
        int lTile = threadIdx.x +threadIdx.y*blockDim.x + i*blockDim.x*blockDim.y;

        if (lTile >= dimTile){continue;}

        int tileX = lTile % (blockDim.x + (K-1));
        int tileY = lTile / (blockDim.x + (K-1));

        int srcX = tileX + blockDim.x * blockIdx.x - center;
        int srcY = tileY + blockDim.y * blockIdx.y - center;

        //linearization
        int src = srcX + srcY * W;

        //copying in shared memory
        for (int c=0; c<C; c++) {
            unsigned char data = 0;

            if (srcX < W && srcY < H && srcX >= 0 && srcY >= 0) {
                data = in[src*C + c]; //because everything is linearized and we have to consider channel stride
            }
            sMem[lTile*C + c] = data;
        }

    }

    __syncthreads();

    if (x >= W || y >= H) {return;}

    for (int c = 0; c < C; c++) {
        float result = 0;
        for (int i = 0; i < K; i++) {
            for (int j = 0; j < K; j++) {
                const int xIdx = j + threadIdx.x;
                const int yIdx = i + threadIdx.y;

                const int inputIdx = (yIdx * (blockDim.x + (K-1)) + xIdx) * C + c;
                result += cuda_kernel[i * K + j] * sMem[inputIdx];
            }
        }
        out[y * W * C + x * C + c] = cudaClamping(result);
    }
}