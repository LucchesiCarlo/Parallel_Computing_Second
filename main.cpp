#include <iostream>
#include "opencv2/core.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "kernel_functions.h"

int main() {
    cv::Mat inputImg = cv::imread("../image.png", cv::IMREAD_UNCHANGED);

    if (inputImg.empty()) {
        std::cerr << "Error: Image not found!" << std::endl;
        return -1;
    }

    cv::Size size = inputImg.size();
    cv::Mat outputImg = cv::Mat::zeros(inputImg.size(), inputImg.type());

    //Access raw bytes of the image
    auto inputPtr = inputImg.ptr();
    auto outputPtr = outputImg.ptr();

    auto kernel = std::unique_ptr<float>(new float[9]);
    for (int i = 0; i < 9; i++) {
        kernel.get()[i] = 1. / 9.;
    }

    applyKernel(inputPtr, outputPtr, kernel.get(), 3, size.width, size.height, inputImg.channels());

    cv::imwrite("../output.png", outputImg);
}