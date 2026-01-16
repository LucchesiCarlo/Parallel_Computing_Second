//
// Created by carlo on 16/01/26.
//
#include <cmath>
#include "kernel_functions.h"


inline unsigned char clamping(float x) {
    if (x < 0.0f) x = 0.0f;
    if (x > 255.0f) x = 255.0f;
    x = std::roundf(x);
    return static_cast<unsigned char> (x);
}

void applyKernel(unsigned char* in, unsigned char* out, float* kernel, int K, int W, int H, int C) {
    int center = K / 2;
    for (int x = 0; x < W; x++) {
        for (int y = 0; y < H; y++) {
            for (int c = 0; c < C; c++) {
                float result = 0;
                for (int i = 0; i < K; i++) {
                    for (int j = 0; j < K; j++) {
                        const int xIdx = i - center + x;
                        const int yIdx = j - center + y;
                        if (!(xIdx < 0 || xIdx >= W || yIdx < 0 || yIdx >= H)) {
                            const int inputIdx = xIdx * C * W + yIdx * C + c;
                            result += kernel[i * K + j] * in[inputIdx];
                        }
                    }
                }
                out[x * W * C + y * C + c] = clamping(result);
            }
        }
    }
}