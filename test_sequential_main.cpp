//
// Created by carlo on 19/01/26.
//
#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;

#include "opencv2/core.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "src/sequential_kernel.h"
#include <omp.h>

int main() {
    float kernel[9];
    generateKernel(kernel, Gaussian);

    std::string inPath = "../input.png";
    std::string outPath = "../output.png";

    cv::Mat inputImg = cv::imread(inPath, cv::IMREAD_UNCHANGED);
    cv::Mat outputImg = cv::Mat::zeros(inputImg.size(), inputImg.type());

    cv::Size size = inputImg.size();

    //Access raw bytes of the image
    auto inputPtr = inputImg.ptr();
    auto outputPtr = outputImg.ptr();

    applyKernel(inputPtr, outputPtr, kernel, 3, size.width, size.height, inputImg.channels());

    cv:imwrite(outPath, outputImg);
}
