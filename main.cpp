#include <iostream>
#include <string>
#include <filesystem>
#include <fstream>
namespace fs = std::filesystem;

#include "opencv2/core.hpp"
#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include "kernel_functions.h"

int main() {

    auto start = std::chrono::high_resolution_clock::now();

    float kernel[9];
    generateKernel(kernel, Gaussian);

    std::string path = "../dataset/seg_pred/seg_pred";
    int count = 0;


    for (const auto & entry : fs::directory_iterator(path)) {
        cv::Mat inputImg = cv::imread(entry.path(), cv::IMREAD_UNCHANGED);
        cv::Mat outputImg = cv::Mat::zeros(inputImg.size(), inputImg.type());

        if (inputImg.empty()) {
            continue;
        }

        cv::Size size = inputImg.size();

        //Access raw bytes of the image
        auto inputPtr = inputImg.ptr();
        auto outputPtr = outputImg.ptr();

        applyKernel(inputPtr, outputPtr, kernel, 3, size.width, size.height, inputImg.channels());
        count++;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto time = std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    std::cout << "Time Taken:" << time << "s" << std::endl;
    std::cout << "Elements Elaborated:" << count << std::endl;

}
