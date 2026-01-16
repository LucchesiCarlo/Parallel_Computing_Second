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

    float kernel[9];
    generateKernel(kernel, Gaussian);

    applyKernel(inputPtr, outputPtr, kernel, 3, size.width, size.height, inputImg.channels());

    cv::imwrite("../output.png", outputImg);
}