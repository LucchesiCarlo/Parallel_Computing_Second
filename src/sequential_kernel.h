//
// Created by carlo on 19/01/26.
//

#ifndef SECOND_ASSIGNMENT_SEQUENTIAL_KERNEL_H
#define SECOND_ASSIGNMENT_SEQUENTIAL_KERNEL_H
#include "kernel_functions.h"

void applyKernel(unsigned char* in, unsigned char* out, float* kernel, int K, int W, int H, int C) {
    int center = K / 2;

    for (int y = 0; y < H; y++){
        for (int x = 0; x < W; x++) {
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
                out[y * W * C + x * C + c] = clamping(result);
            }
        }
    }
}
#endif //SECOND_ASSIGNMENT_SEQUENTIAL_KERNEL_H