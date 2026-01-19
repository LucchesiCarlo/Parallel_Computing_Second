//
// Created by carlo on 19/01/26.
//

#ifndef SECOND_ASSIGNMENT_KERNEL_FUNCTIONS_H
#define SECOND_ASSIGNMENT_KERNEL_FUNCTIONS_H

enum KernelType {
    Gaussian,
    Ridge,
    Sharpen,
    Identity
};

inline void generateKernel(float *kernel, KernelType type) {
    switch (type) {
        case Gaussian:
            for (int i = 0; i < 9; i++) {
                kernel[i] = 1.f / 9.f;
            }
            break;
        case Ridge:
            for (int i = 0; i < 9; i++) {
                kernel[i] = -1;
            }
            kernel[4] = 8;
        case Sharpen:
            for (int i = 0; i < 9; i++) {
                kernel[i] = 0;
            }
            kernel[4] = 5;
            kernel[1] = -1;
            kernel[3] = -1;
            kernel[5] = -1;
            kernel[7] = -1;
            break;
        case Identity:
        default:
            for (int i = 0; i < 9; i++) {
                kernel[i] = 0;
            }
            kernel[4] = 1;
    }
}

inline unsigned char clamping(float x) {
    if (x < 0.0f) x = 0.0f;
    if (x > 255.0f) x = 255.0f;
    x = std::roundf(x);
    return static_cast<unsigned char> (x);
}

#endif //SECOND_ASSIGNMENT_KERNEL_FUNCTIONS_H